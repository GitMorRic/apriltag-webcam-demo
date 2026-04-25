"""
training/demo_pc/demo.py
PC 端 AprilTag 实时识别演示（带可视化窗口 + 录制 + 低通滤波）

功能：
  - 使用笔记本/USB 摄像头实时识别 AprilTag（Tag36h11）
  - 复用 vision_raspberrypi/src/ 的 AprilTagDetector + Strategy
  - 复用 shared/visualization.py 的可视化工具
  - 姿态低通滤波：抑制 PnP 解算抖动（alpha 可调）
  - 截图 / 录制视频：便于调试复盘
  - 弹出窗口实时显示：
      · 每个 tag 的边框、3D 坐标轴、ID、距离、偏角
      · 顶部 HUD：当前 ACTION / 检测数量 / FPS
      · 右侧详情面板：所有检测 tag 的完整信息
      · 右下角俯视小地图：tag 相对于车的位置
      · 底部状态条：目标信息 / 控制指令 / reason
  - 终端打印决策结果（按帧节流，避免刷屏）

运行方式：
  cd WalkingRobotVision
  python training/demo_pc/demo.py
  python training/demo_pc/demo.py --camera 1            # 用第 2 个摄像头
  python training/demo_pc/demo.py --tag-size 0.16       # 自定义 tag 边长
  python training/demo_pc/demo.py --no-minimap          # 关闭小地图
  python training/demo_pc/demo.py --no-info-panel       # 关闭右侧面板
  python training/demo_pc/demo.py --alpha 0.5           # 低通滤波强度（0~1，越小越平滑）

按键：
  q / Esc      退出
  s            保存截图到 recordings/ 目录（含时间戳）
  r            开始 / 停止录制视频（保存到 recordings/）
  空格         暂停 / 继续
  m            切换鸟瞰小地图显示
  i            切换右侧详情面板显示
  h            切换 HUD 显示
  f            切换姿态低通滤波（开/关，用于对比效果）
"""

import sys
import os
import time
import logging
import argparse
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# ── 路径设置：让 Python 能找到 shared/ 和 vision_raspberrypi/src/ ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vision_raspberrypi"))

from vision_raspberrypi.src.apriltag_detector import AprilTagDetector
from vision_raspberrypi.src.strategy import Strategy, ActionResult

# 复用通用可视化模块（同样可以被树莓派/调试工具使用）
from shared import visualization as viz

from camera_opencv import OpenCVCamera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo_pc")


# ────────────────── 默认配置（PC 演示用） ──────────────────
# 笔记本摄像头粗略估算值（640×480，FOV ~60°）
# 距离测量会有 10-20% 误差，但识别和姿态估计完全正常
DEFAULT_CALIB = {
    "fx": 640.0,
    "fy": 640.0,
    "cx": 320.0,
    "cy": 240.0,
    "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
}

DEFAULT_DETECT = {
    "family":        "tag36h11",
    "tag_size":      0.12,    # 比赛用 tag 边长，米
    "nthreads":      2,
    "quad_decimate": 1.0,     # PC 上 CPU 快，不需要降采样
    "quad_sigma":    0.0,
    "refine_edges":  True,
}

DEFAULT_STRATEGY = {
    "our_team_id":   1,
    "enemy_team_id": 2,
    "distances": {
        "push_dist":     0.30,
        "approach_dist": 1.20,
        "avoid_dist":    0.40,
    },
    "speeds": {
        "approach": 40.0,
        "push":     60.0,
        "turn":     35.0,
        "search":  -25.0,
    },
}


# ────────────────── 工具类 ──────────────────

class FpsMeter:
    """滑动窗口 FPS 计算（最近 30 帧）。"""
    def __init__(self, window: int = 30):
        self._t = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self._t.append(now)
        if len(self._t) < 2:
            return 0.0
        dt = self._t[-1] - self._t[0]
        return (len(self._t) - 1) / dt if dt > 0 else 0.0


class PoseSmoother:
    """
    对检测到的 pose（距离 / yaw / pitch / tvec）做一阶低通滤波。

    为什么要滤波：
      PnP 解算对相机内参非常敏感。使用估算内参（fx=fy=640）时，
      角点检测的 1~2 像素噪声会被放大成数十度的姿态跳变（坐标轴乱抖）
      和距离估算误差。低通滤波不能修正内参误差，但能让可视化更稳定，
      便于在无法做标定的场合（培训演示）观察相对变化趋势。

    正确做法（生产环境）：做完整的相机标定（见 vision_raspberrypi/tools/calibrate.py）。

    alpha：每帧更新权重，范围 0~1
      alpha=1.0 → 不滤波（原始值）
      alpha=0.3 → 强平滑（响应慢，适合静态展示）
      alpha=0.6 → 中等平滑（默认）
    """
    def __init__(self, alpha: float = 0.6):
        self.alpha  = alpha
        self._state: dict = {}   # tag_id → {dist, yaw, pitch, tx, ty, tz}

    def smooth(self, detections: list) -> list:
        """对 detections 列表里的每个 DetectionResult 做原地平滑，返回同一列表。

        同时平滑：
          - pose_t（平移向量）→ 控制小地图上的位置点
          - pose_R（旋转矩阵，通过 rvec 线性插值）→ 控制坐标轴箭头方向
          - distance / yaw_deg / pitch_deg → 控制文字标签数值
        """
        if self.alpha >= 1.0:
            return detections   # alpha=1 时跳过（不改任何值）

        seen = set()
        for det in detections:
            tid = det.tag_id
            seen.add(tid)

            # 当前帧的旋转向量（轴角）
            rvec_now, _ = cv2.Rodrigues(det.pose_R)
            rvec_now    = rvec_now.flatten()

            if tid not in self._state:
                self._state[tid] = {
                    "dist":  det.distance,
                    "yaw":   det.yaw_deg,
                    "pitch": det.pitch_deg,
                    "tx":    float(det.pose_t[0]),
                    "ty":    float(det.pose_t[1]),
                    "tz":    float(det.pose_t[2]),
                    "rvec":  rvec_now.copy(),
                }
            else:
                s = self._state[tid]
                a = self.alpha

                # 平移 & 数值
                s["dist"]  = a * det.distance  + (1 - a) * s["dist"]
                s["yaw"]   = a * det.yaw_deg   + (1 - a) * s["yaw"]
                s["pitch"] = a * det.pitch_deg + (1 - a) * s["pitch"]
                s["tx"]    = a * det.pose_t[0] + (1 - a) * s["tx"]
                s["ty"]    = a * det.pose_t[1] + (1 - a) * s["ty"]
                s["tz"]    = a * det.pose_t[2] + (1 - a) * s["tz"]

                # 旋转：对 rvec（轴角向量）做线性插值，再转回旋转矩阵
                # 注意：rvec 轴角插值在旋转角接近 π 时可能有不连续，
                # 但对于大多数正面朝前的场景已足够稳定。
                rvec_prev = s["rvec"]
                # 如果两个旋转向量方向反号（同一旋转的两种表示），取近的那个
                if np.dot(rvec_now, rvec_prev) < 0:
                    rvec_now = -rvec_now
                s["rvec"]  = a * rvec_now + (1 - a) * rvec_prev

                # 写回
                det.distance  = s["dist"]
                det.yaw_deg   = s["yaw"]
                det.pitch_deg = s["pitch"]
                det.pose_t    = np.array([s["tx"], s["ty"], s["tz"]])
                smoothed_R, _ = cv2.Rodrigues(s["rvec"].reshape(3, 1))
                det.pose_R    = smoothed_R

        for tid in list(self._state.keys()):
            if tid not in seen:
                del self._state[tid]

        return detections


class VideoRecorder:
    """封装 cv2.VideoWriter，支持开始/停止录制。"""
    def __init__(self, save_dir: str, fps: float, width: int, height: int):
        self._dir    = save_dir
        self._fps    = fps
        self._w      = width
        self._h      = height
        self._writer = None
        self._path   = None

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def start(self) -> str:
        os.makedirs(self._dir, exist_ok=True)
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        path  = os.path.join(self._dir, f"rec_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, self._fps,
                                       (self._w, self._h))
        self._path = path
        return path

    def write(self, frame: np.ndarray):
        if self._writer is not None:
            self._writer.write(frame)

    def stop(self) -> str:
        path = self._path
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            self._path   = None
        return path


# ────────────────── 主流程 ──────────────────

def run(args):
    cam = OpenCVCamera(device_id=args.camera, width=args.width, height=args.height)
    if not cam.open():
        log.error("摄像头打开失败（device_id=%d），退出", args.camera)
        log.error("提示：用 --camera 1/2/... 试其他设备 ID；")
        log.error("      Win 上虚拟摄像头/被占用都会失败。")
        return

    detect_cfg = dict(DEFAULT_DETECT)
    if args.tag_size:
        detect_cfg["tag_size"] = args.tag_size

    strategy_cfg = dict(DEFAULT_STRATEGY)
    if args.our_team:
        strategy_cfg["our_team_id"]   = args.our_team
        strategy_cfg["enemy_team_id"] = 3 - args.our_team

    detector = AprilTagDetector(DEFAULT_CALIB, detect_cfg)
    engine   = Strategy(strategy_cfg)
    smoother = PoseSmoother(alpha=args.alpha)

    # 录制目录：项目根目录下的 recordings/
    rec_dir  = os.path.join(ROOT, "recordings")
    recorder = VideoRecorder(rec_dir, fps=25.0, width=args.width, height=args.height)

    log.info("演示启动 —— 按 q 退出 / s 截图 / r 录制 / 空格暂停 / m i h f 切换面板")
    log.info("► 使用估算内参（fx=fy=640），距离/姿态误差较大，需做标定才能精确")
    log.info("► 低通滤波 alpha=%.2f（按 f 切换，1.0=关闭）", args.alpha)
    log.info("► 我方=ID%d  敌方=ID%d  中立=ID0  tag_size=%.3fm",
             strategy_cfg["our_team_id"], strategy_cfg["enemy_team_id"],
             detect_cfg["tag_size"])

    # 准备渲染参数
    cam_mtx = np.array([[DEFAULT_CALIB["fx"], 0, DEFAULT_CALIB["cx"]],
                        [0, DEFAULT_CALIB["fy"], DEFAULT_CALIB["cy"]],
                        [0, 0, 1]], dtype=np.float64)
    dist    = np.array(DEFAULT_CALIB["dist_coeffs"], dtype=np.float64)

    show_minimap    = not args.no_minimap
    show_info_panel = not args.no_info_panel
    show_hud        = True
    filter_on       = True     # 低通滤波开关
    paused          = False
    frame_id        = 0
    last_log_t      = 0.0
    fps_meter       = FpsMeter()
    frozen_frame    = None

    win_title = ("AprilTag PC Demo  "
                 "[q=quit  s=screenshot  r=record  space=pause  m=map  i=info  h=hud  f=filter]")
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, args.width, args.height)

    try:
        while True:
            if not paused:
                ok, frame = cam.read()
                if not ok:
                    log.warning("读帧失败，跳过")
                    time.sleep(0.05)
                    continue

                raw_frame  = frame.copy()        # 原始帧，用于录制（叠加可视化后再写）
                detections = detector.detect(frame)

                # 姿态低通滤波（原地修改 detections 的 pose 字段）
                if filter_on:
                    detections = smoother.smooth(detections)

                action = engine.decide(detections)
                fps    = fps_meter.tick()

                # ── 一站式可视化叠加 ──
                if show_hud:
                    viz.render_overlay(
                        frame, detections, action,
                        camera_matrix=cam_mtx, dist_coeffs=dist,
                        tag_size_m=detect_cfg["tag_size"],
                        fps=fps,
                        our_team_id=strategy_cfg["our_team_id"],
                        enemy_team_id=strategy_cfg["enemy_team_id"],
                        show_minimap=show_minimap,
                        show_info_panel=show_info_panel,
                    )
                else:
                    viz.draw_all_detections(
                        frame, detections, cam_mtx, dist,
                        detect_cfg["tag_size"],
                        target_tag_id=(action.tag_id
                                       if action.tag_id not in (0xFF, 255) else None),
                    )

                # 估算内参警告水印（右上角）
                _draw_calib_warning(frame, filter_on, smoother.alpha)

                # 录制状态指示
                if recorder.is_recording:
                    _draw_rec_indicator(frame, frame_id)
                    recorder.write(frame)

                # 终端打印（每 0.5s 一次）
                now = time.perf_counter()
                if now - last_log_t > 0.5:
                    last_log_t = now
                    if detections:
                        r = detections[0]
                        log.info(
                            "tag_id=%d dist=%.2fm yaw=%+.1f° pitch=%+.1f° "
                            "→ action=%s v=%+.0f%% ω=%+.0f%%",
                            r.tag_id, r.distance, r.yaw_deg, r.pitch_deg,
                            viz.DEFAULT_ACTION_NAMES.get(action.action, "?"),
                            action.v_pct, action.omega_pct,
                        )
                    else:
                        log.info("未检测到 tag → action=%s",
                                 viz.DEFAULT_ACTION_NAMES.get(action.action, "?"))

                frozen_frame = frame.copy()
                frame_id += 1
            else:
                frame = frozen_frame.copy() if frozen_frame is not None else \
                        np.zeros((args.height, args.width, 3), dtype=np.uint8)
                viz.put_text(frame, "PAUSED", (frame.shape[1] // 2 - 60, 80),
                             viz.COLOR_YELLOW, scale=1.2, thickness=3)

            cv2.imshow(win_title, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                _save_screenshot(frame, rec_dir, frame_id)
            elif key == ord('r'):
                if recorder.is_recording:
                    path = recorder.stop()
                    log.info("录制停止 → %s", path)
                else:
                    path = recorder.start()
                    log.info("开始录制 → %s", path)
            elif key == ord(' '):
                paused = not paused
                log.info("[%s]", "暂停" if paused else "继续")
            elif key == ord('m'):
                show_minimap = not show_minimap
                log.info("小地图: %s", "ON" if show_minimap else "OFF")
            elif key == ord('i'):
                show_info_panel = not show_info_panel
                log.info("详情面板: %s", "ON" if show_info_panel else "OFF")
            elif key == ord('h'):
                show_hud = not show_hud
                log.info("全部 HUD: %s", "ON" if show_hud else "OFF")
            elif key == ord('f'):
                filter_on = not filter_on
                smoother._state.clear()   # 切换时清空历史避免残留
                log.info("低通滤波: %s (alpha=%.2f)", "ON" if filter_on else "OFF",
                         smoother.alpha)

    except KeyboardInterrupt:
        log.info("收到 Ctrl+C")
    finally:
        if recorder.is_recording:
            path = recorder.stop()
            log.info("录制自动保存 → %s", path)
        cam.close()
        cv2.destroyAllWindows()
        log.info("演示结束（共渲染 %d 帧）", frame_id)


# ────────────────── 辅助绘制 ──────────────────

def _draw_calib_warning(frame: np.ndarray, filter_on: bool, alpha: float):
    """在右上角显示内参警告和滤波状态，提醒距离/姿态数值仅供参考。"""
    h, w = frame.shape[:2]
    lines = [
        ("CALIB: ESTIMATED", viz.COLOR_ORANGE),
        ("dist/pose approx", viz.COLOR_ORANGE),
        (f"FILTER: {'ON a={:.2f}'.format(alpha) if filter_on else 'OFF'}",
         viz.COLOR_GREEN if filter_on else viz.COLOR_GRAY),
    ]
    for i, (text, color) in enumerate(lines):
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        viz.put_text(frame, text, (w - tw - 8, 60 + i * 16),
                     color=color, scale=0.38, thickness=1, shadow=True)


def _draw_rec_indicator(frame: np.ndarray, frame_id: int):
    """录制中时在左上角画红色 REC 指示（每秒闪烁）。"""
    if (frame_id // 15) % 2 == 0:   # 每 15 帧切换一次（约 0.5s）
        cv2.circle(frame, (18, 62), 8, viz.COLOR_RED, -1, cv2.LINE_AA)
        viz.put_text(frame, "REC", (30, 68), viz.COLOR_RED,
                     scale=0.6, thickness=2, shadow=True)


def _save_screenshot(frame: np.ndarray, save_dir: str, frame_id: int) -> str:
    os.makedirs(save_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"screenshot_{ts}_{frame_id:06d}.jpg")
    cv2.imwrite(path, frame)
    log.info("截图已保存：%s", path)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="AprilTag PC 端实时识别演示（复用项目生产代码）",
    )
    parser.add_argument("--camera",   type=int, default=0,
                        help="摄像头设备 ID（默认 0 = 笔记本内置摄像头）")
    parser.add_argument("--width",    type=int, default=640,
                        help="采集分辨率宽度（默认 640）")
    parser.add_argument("--height",   type=int, default=480,
                        help="采集分辨率高度（默认 480）")
    parser.add_argument("--tag-size", type=float, default=None,
                        help="Tag 物理边长（米，默认 0.12 = 比赛用 12cm）")
    parser.add_argument("--our-team", type=int, default=1, choices=[1, 2],
                        help="我方队伍 ID（1=蓝，2=黄，默认 1）")
    parser.add_argument("--no-minimap",    action="store_true",
                        help="启动时关闭右下角小地图")
    parser.add_argument("--no-info-panel", action="store_true",
                        help="启动时关闭右侧详情面板")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="姿态低通滤波系数 0~1（越小越平滑，1.0=关闭，默认 0.6）")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
