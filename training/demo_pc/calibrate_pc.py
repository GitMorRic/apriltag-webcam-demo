#!/usr/bin/env python3
"""
training/demo_pc/calibrate_pc.py
PC 端相机标定工具（用笔记本/USB 摄像头 + 棋盘格）

用法：
    cd WalkingRobotVision
    python training/demo_pc/calibrate_pc.py

    python training/demo_pc/calibrate_pc.py --camera 1    # 第 2 个摄像头
    python training/demo_pc/calibrate_pc.py \\
        --rows 9 --cols 6 --square 0.025 \\             # 9行6列格子25mm
        --n-frames 25 \\                                # 采集25张
        --output shared/calibration/my_webcam_640x480.yaml

操作方法：
    1. 打印 9行6列（内角点）棋盘格（格边长 25mm），贴在硬纸板上压平
    2. 运行此脚本，弹出摄像头窗口
    3. 将棋盘格放入视野，检测到时角点变绿 → 按 空格 采集一张
    4. 换不同角度/位置再按 空格（建议覆盖图像四角，倾斜角 15~30°）
    5. 采集到目标帧数后自动计算并保存 YAML

按键：
    空格    当检测到棋盘格时采集当前帧
    d       删除最后一张采集的图（采错了可以撤销）
    q       提前结束采集并直接计算（已有图片 >= 10 张时）
    ESC     强制退出（不保存）

棋盘格说明：
    内角点数 = 格子数 - 1
    本脚本默认：9行×6列内角点，对应 10×7 格子的棋盘
    格边长默认：25mm（务必用实际打印出来的值！用尺子量！）
"""

import sys
import os
import time
import logging
import argparse

import cv2
import numpy as np
import yaml

# ── 路径设置 ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("calibrate_pc")


def run(args):
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        log.error("无法打开摄像头 %d", args.camera)
        sys.exit(1)

    rows, cols = args.rows, args.cols
    square_m   = args.square

    # 棋盘角点在真实世界的 3D 坐标（Z=0 平面）
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_pts: list = []     # 每张图的 3D 角点坐标
    img_pts: list = []     # 每张图的 2D 角点像素坐标
    img_size = (args.width, args.height)

    log.info("目标：%d 张有效图。检测到棋盘格后按 空格 采集，q 提前完成，ESC 退出。", args.n_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        vis = frame.copy()

        if found:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis, (cols, rows), refined, found)
            msg = f"FOUND  Space to capture  [{len(obj_pts)}/{args.n_frames}]"
            cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)
        else:
            cv2.putText(vis, f"Not found  [{len(obj_pts)}/{args.n_frames}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 255), 2)

        # 采集进度条
        bar_w = int(args.width * len(obj_pts) / args.n_frames)
        cv2.rectangle(vis, (0, args.height - 8), (bar_w, args.height),
                      (0, 200, 100), -1)

        cv2.imshow("Camera Calibration  [Space=capture  d=undo  q=done  ESC=abort]", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:                                  # ESC
            log.info("用户中止，未保存")
            cap.release()
            cv2.destroyAllWindows()
            return

        elif key == ord(' ') and found:                # 空格：采集
            obj_pts.append(objp.copy())
            img_pts.append(refined)
            log.info("已采集 %d / %d", len(obj_pts), args.n_frames)
            if len(obj_pts) >= args.n_frames:
                break

        elif key == ord('d') and obj_pts:              # d：撤销最后一张
            obj_pts.pop()
            img_pts.pop()
            log.info("已删除，剩余 %d 张", len(obj_pts))

        elif key == ord('q') and len(obj_pts) >= 10:   # q：提前完成
            log.info("提前完成，已有 %d 张", len(obj_pts))
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(obj_pts) < 10:
        log.error("有效帧数不足（%d），无法标定。建议重新采集。", len(obj_pts))
        sys.exit(1)

    # ─── 标定 ───
    log.info("正在计算内参（%d 张图）…", len(obj_pts))
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None)

    log.info("标定完成！RMS 重投影误差 = %.4f 像素", rms)
    if rms < 0.5:
        log.info("✓ 优秀（< 0.5px）")
    elif rms < 1.0:
        log.info("✓ 合格（< 1.0px）")
    else:
        log.warning("⚠ 误差偏大（> 1.0px），建议检查棋盘格是否平整，或增加采集张数")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    d      = [round(float(v), 8) for v in dist.flatten()[:5]]

    log.info("内参：fx=%.2f  fy=%.2f  cx=%.2f  cy=%.2f", fx, fy, cx, cy)
    log.info("畸变：%s", d)

    # ─── 可视化验证：在窗口里显示去畸变效果 ───
    log.info("按任意键查看去畸变效果（ESC 跳过）…")
    cap2 = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, img_size, 1, img_size)
    for _ in range(60):          # 最多显示 2 秒
        ret, fr = cap2.read()
        if not ret:
            break
        undist = cv2.undistort(fr, K, dist, None, newK)
        combined = np.hstack([fr, undist])
        cv2.putText(combined, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(combined, "Undistorted", (fr.shape[1] + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Undistort preview  [any key to continue]", combined)
        if cv2.waitKey(30) != -1:
            break
    cap2.release()
    cv2.destroyAllWindows()

    # ─── 保存 YAML ───
    out_path = os.path.join(ROOT, args.output) \
               if not os.path.isabs(args.output) else args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    result = {
        "camera": {
            "resolution": list(img_size),
            "model":      f"webcam_camera{args.camera}",
        },
        "intrinsics": {
            "fx": round(fx, 4),
            "fy": round(fy, 4),
            "cx": round(cx, 4),
            "cy": round(cy, 4),
        },
        "distortion": {
            "coeffs": d,
            "comment": "[k1, k2, p1, p2, k3]",
        },
        "calibration_date": time.strftime("%Y-%m-%d"),
        "calibration_tool": f"calibrate_pc.py checkerboard {rows}x{cols} {square_m*1000:.0f}mm",
        "rms_error": round(rms, 4),
    }

    with open(out_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
    log.info("已保存到：%s", out_path)

    # ─── 打印"填写指南" ───
    print("\n" + "=" * 60)
    print("标定完成！请将以下值填写到对应配置文件：")
    print("=" * 60)
    print(f"\n【training/demo_pc/demo.py → DEFAULT_CALIB】")
    print(f'    "fx": {round(fx,2)},')
    print(f'    "fy": {round(fy,2)},')
    print(f'    "cx": {round(cx,2)},')
    print(f'    "cy": {round(cy,2)},')
    print(f'    "dist_coeffs": {d},')
    print(f"\n【vision_raspberrypi/config.yaml → calibration】")
    print(f"    fx:   {round(fx,2)}")
    print(f"    fy:   {round(fy,2)}")
    print(f"    cx:   {round(cx,2)}")
    print(f"    cy:   {round(cy,2)}")
    print(f"    dist_coeffs: {d}")
    print(f"\n完整 YAML 文件：{out_path}")
    print(f"RMS 误差：{round(rms, 4)} 像素")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="PC 端棋盘格相机标定（用笔记本/USB 摄像头）")
    parser.add_argument("--camera",   type=int, default=0,
                        help="摄像头设备 ID（默认 0）")
    parser.add_argument("--width",    type=int, default=640)
    parser.add_argument("--height",   type=int, default=480)
    parser.add_argument("--rows",     type=int, default=9,
                        help="棋盘格内角点行数（默认 9）")
    parser.add_argument("--cols",     type=int, default=6,
                        help="棋盘格内角点列数（默认 6）")
    parser.add_argument("--square",   type=float, default=0.025,
                        help="格子实际边长（米），默认 0.025 = 25mm，务必用尺子量！")
    parser.add_argument("--n-frames", type=int, default=25,
                        help="目标采集帧数（默认 25）")
    parser.add_argument("--output",   default="shared/calibration/my_webcam_640x480.yaml",
                        help="输出 YAML 路径（相对于项目根目录）")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
