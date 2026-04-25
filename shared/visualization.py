"""
shared/visualization.py
通用可视化工具 —— PC demo / 树莓派调试 / 任意 OpenCV 流程都可复用。

设计原则：
  - 纯 OpenCV 实现，零额外依赖（不依赖 PIL / matplotlib）
  - 所有文字均为英文（cv2.putText 不支持中文，避免引入字体文件）
  - 函数式 API，每个 draw_* 直接修改 frame，并返回它

主要 API：
  draw_detection_overlay(frame, det, ...)  → 单个 tag 的框 + 角点 + 坐标轴 + 标签
  draw_all_detections(frame, dets, ...)    → 批量画所有 tag
  draw_top_hud(frame, ...)                 → 顶部半透明状态条（ACTION / FPS / TARGET）
  draw_info_panel(frame, dets, ...)        → 右侧详情面板（每个 tag 的详细信息）
  draw_minimap(frame, dets, ...)           → 右下角俯视小地图
  draw_bottom_bar(frame, action_result)    → 底部状态条（reason / v / omega）
  draw_target_highlight(frame, det)        → 高亮当前选中目标
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


# ──────────────────────────── 常量 ────────────────────────────

# 颜色（BGR）
COLOR_GREEN   = (0, 255, 0)
COLOR_RED     = (0, 0, 255)
COLOR_BLUE    = (255, 0, 0)
COLOR_YELLOW  = (0, 255, 255)
COLOR_CYAN    = (255, 255, 0)
COLOR_ORANGE  = (0, 165, 255)
COLOR_GRAY    = (128, 128, 128)
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0, 0, 0)
COLOR_DARK    = (40, 40, 40)

# Tag ID → 阵营颜色（按比赛规则：1=蓝方，2=黄方，0=中立）
TAG_TEAM_COLORS = {
    0: (200, 200, 200),   # 中立 - 浅灰
    1: (255, 80, 80),     # 蓝方 - 蓝（注意 OpenCV 是 BGR）
    2: (80, 220, 255),    # 黄方 - 亮黄
}

TAG_TEAM_NAMES = {
    0: "NEUTRAL",
    1: "BLUE",
    2: "YELLOW",
}

# 默认动作名称（与 protocol.h 的 ACTION_* 对应）
DEFAULT_ACTION_NAMES = {
    0: "STOP",
    1: "VELOCITY",
    2: "PUSH",
    3: "AVOID",
    4: "PROTECT",
    5: "SEARCH",
}

DEFAULT_ACTION_COLORS = {
    0: COLOR_GRAY,        # STOP
    1: COLOR_CYAN,        # VELOCITY
    2: COLOR_GREEN,       # PUSH
    3: COLOR_ORANGE,      # AVOID
    4: COLOR_BLUE,        # PROTECT
    5: COLOR_YELLOW,      # SEARCH
}


# ──────────────────────────── 内部工具 ────────────────────────────

def _alpha_rect(frame: np.ndarray,
                top_left: Tuple[int, int],
                bot_right: Tuple[int, int],
                color: Tuple[int, int, int],
                alpha: float = 0.5) -> None:
    """在 frame 上绘制半透明实心矩形（in-place）。"""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bot_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def put_text(frame: np.ndarray, text: str, org: Tuple[int, int],
             color: Tuple[int, int, int] = COLOR_WHITE,
             scale: float = 0.5, thickness: int = 1,
             shadow: bool = True) -> None:
    """带可选阴影的文字（提高可读性，公开 API）。"""
    if shadow:
        cv2.putText(frame, text, (org[0] + 1, org[1] + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, COLOR_BLACK, thickness + 1,
                    cv2.LINE_AA)
    cv2.putText(frame, text, org,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


# 内部别名（保持下面已经使用 _put_text 的代码不变）
_put_text = put_text


def _team_color(tag_id: int) -> Tuple[int, int, int]:
    return TAG_TEAM_COLORS.get(tag_id, COLOR_WHITE)


def _team_name(tag_id: int) -> str:
    return TAG_TEAM_NAMES.get(tag_id, f"TAG{tag_id}")


# ──────────────────────────── 检测可视化 ────────────────────────────

def draw_detection_overlay(frame: np.ndarray, det,
                           camera_matrix: np.ndarray,
                           dist_coeffs: np.ndarray,
                           tag_size_m: float,
                           is_target: bool = False) -> np.ndarray:
    """
    在 frame 上绘制单个 AprilTag 检测结果（边框 + 角点 + 3D 坐标轴 + 标签）。

    Args:
        frame:          BGR 图像
        det:            DetectionResult（含 tag_id / corners / center / pose_R / pose_t）
        camera_matrix:  3×3 相机内参（用于投影坐标轴）
        dist_coeffs:    畸变系数
        tag_size_m:     tag 物理边长（米）
        is_target:      是否是当前选中目标（高亮）

    Returns:
        frame（已修改）
    """
    tag_color = _team_color(det.tag_id)
    border_thick = 4 if is_target else 2

    # 1) 四边框
    pts = det.corners.astype(int)
    cv2.polylines(frame, [pts], True, tag_color, border_thick, cv2.LINE_AA)

    # 2) 四个角点小圆点（红绿蓝白：左下、右下、右上、左上）
    corner_palette = [COLOR_RED, COLOR_GREEN, COLOR_BLUE, COLOR_WHITE]
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (int(x), int(y)), 4, corner_palette[i], -1, cv2.LINE_AA)

    # 3) 中心点
    cx, cy = int(det.center[0]), int(det.center[1])
    cv2.circle(frame, (cx, cy), 6, tag_color, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, COLOR_WHITE, -1, cv2.LINE_AA)

    # 4) 3D 坐标轴（X 红、Y 绿、Z 蓝）
    _draw_axes_3d(frame, det, camera_matrix, dist_coeffs, tag_size_m)

    # 5) 顶上贴标签（ID + 队伍 + 距离 + 偏角）
    label = f"ID={det.tag_id} {_team_name(det.tag_id)}"
    info  = f"D={det.distance:.2f}m  Y={det.yaw_deg:+.1f}d"
    _draw_tag_label(frame, pts, label, info, tag_color, is_target)

    # 6) 目标高亮——画一个外圈
    if is_target:
        radius = max(8, int(np.linalg.norm(pts[0] - pts[2]) / 2 + 12))
        cv2.circle(frame, (cx, cy), radius, COLOR_YELLOW, 2, cv2.LINE_AA)

    return frame


def _draw_axes_3d(frame, det, camera_matrix, dist_coeffs, tag_size):
    """画 tag 的 3D 坐标轴（X 红 / Y 绿 / Z 蓝）。

    会先做 PnP 健康检查：
      - tvec[2]（深度）必须落在 5cm ~ 5m 区间
      - 投影后的端点不能离 tag 中心太远（>2 倍画面尺寸视为爆掉）
    检查失败时（典型：内参不准 + tag_size 不匹配 → 极近 dist），
    自动 fallback 到基于角点的 2D"伪坐标轴"，避免出现穿屏的长箭头。
    """
    h, w = frame.shape[:2]
    cx_img, cy_img = int(det.center[0]), int(det.center[1])

    tz = float(det.pose_t[2])
    # 1) 距离合理性
    if 0.05 < abs(tz) < 5.0:
        axis_len = tag_size * 0.6
        axes = np.array([[0, 0, 0],
                         [axis_len, 0, 0],
                         [0, axis_len, 0],
                         [0, 0, -axis_len]], dtype=np.float32)
        rvec, _ = cv2.Rodrigues(det.pose_R)
        pts, _ = cv2.projectPoints(axes, rvec, det.pose_t.reshape(3, 1),
                                   camera_matrix, dist_coeffs)
        pts = pts.reshape(-1, 2).astype(int)

        # 2) 投影端点合理性：离 tag 中心不能超过 2 倍画面尺寸
        max_extent = max(w, h) * 2
        if all(abs(int(p[0]) - cx_img) < max_extent and
               abs(int(p[1]) - cy_img) < max_extent for p in pts):
            o = tuple(pts[0])
            cv2.arrowedLine(frame, o, tuple(pts[1]), COLOR_RED,
                            2, cv2.LINE_AA, tipLength=0.3)
            cv2.arrowedLine(frame, o, tuple(pts[2]), COLOR_GREEN,
                            2, cv2.LINE_AA, tipLength=0.3)
            cv2.arrowedLine(frame, o, tuple(pts[3]), COLOR_BLUE,
                            2, cv2.LINE_AA, tipLength=0.3)
            return

    # 3) Fallback：PnP 不可信 → 基于角点画"伪坐标系"
    _draw_pseudo_axes_2d(frame, det)


def _draw_pseudo_axes_2d(frame, det):
    """从 tag 中心，沿 tag 平面内 X/Y 方向画两根短箭头。

    - 红色 X 轴：从中心指向 tag 底边右侧（沿 LR-LL 方向）
    - 绿色 Y 轴：从中心指向 tag 左边上方（沿 UL-LL 方向）
    - 蓝色 Z 轴：在中心画一个空心圈表示"朝外"（无 PnP 时给不出长度）

    用途：内参未标定或 PnP 距离异常时的备用可视化，
    箭头长度按 tag 边长比例计算，永远不会"飞出屏幕"。
    """
    pts = det.corners.astype(np.float32)   # [LL, LR, UR, UL]（apriltag 输出顺序）
    cx, cy = float(det.center[0]), float(det.center[1])

    # 沿 tag 平面内的 X / Y 方向各取 40% 长度
    x_dir = (pts[1] - pts[0]) * 0.4        # LR - LL
    y_dir = (pts[3] - pts[0]) * 0.4        # UL - LL

    o   = (int(cx), int(cy))
    x_e = (int(cx + x_dir[0]), int(cy + x_dir[1]))
    y_e = (int(cx + y_dir[0]), int(cy + y_dir[1]))

    cv2.arrowedLine(frame, o, x_e, COLOR_RED,   2, cv2.LINE_AA, tipLength=0.3)
    cv2.arrowedLine(frame, o, y_e, COLOR_GREEN, 2, cv2.LINE_AA, tipLength=0.3)

    # Z 轴：用一个空心圈+实心点表示"朝向相机"（无标定算不出真实方向）
    cv2.circle(frame, o, 6, COLOR_BLUE, 2, cv2.LINE_AA)
    cv2.circle(frame, o, 2, COLOR_BLUE, -1, cv2.LINE_AA)

    # 小标注，提醒用户"这个不是真 PnP 姿态"
    _put_text(frame, "no-pnp", (o[0] + 8, o[1] + 18),
              COLOR_GRAY, scale=0.35, thickness=1)


def _draw_tag_label(frame: np.ndarray, pts: np.ndarray,
                    label_top: str, label_bot: str,
                    color: Tuple[int, int, int], is_target: bool):
    """在四边形上方贴一个两行的标签框。"""
    x_min = int(pts[:, 0].min())
    y_min = int(pts[:, 1].min())

    # 背景框尺寸
    pad = 4
    w_top = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    w_bot = cv2.getTextSize(label_bot, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
    box_w = max(w_top, w_bot) + pad * 2
    box_h = 36

    # 默认贴在四边形上方
    bx0, by0 = x_min, y_min - box_h - 4
    if by0 < 2:                          # 上方放不下，贴下方
        by0 = int(pts[:, 1].max()) + 4
    bx1, by1 = bx0 + box_w, by0 + box_h

    _alpha_rect(frame, (bx0, by0), (bx1, by1), COLOR_BLACK, alpha=0.65)
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), color, 2 if is_target else 1)

    _put_text(frame, label_top, (bx0 + pad, by0 + 14),
              color=color, scale=0.5, thickness=1)
    _put_text(frame, label_bot, (bx0 + pad, by0 + 30),
              color=COLOR_WHITE, scale=0.45, thickness=1)


def draw_all_detections(frame: np.ndarray, detections: List,
                        camera_matrix: np.ndarray,
                        dist_coeffs: np.ndarray,
                        tag_size_m: float,
                        target_tag_id: Optional[int] = None) -> np.ndarray:
    """批量绘制所有检测结果，target_tag_id 对应的会被高亮。"""
    for det in detections:
        is_target = (target_tag_id is not None and det.tag_id == target_tag_id)
        draw_detection_overlay(frame, det, camera_matrix, dist_coeffs,
                                tag_size_m, is_target=is_target)
    return frame


# ──────────────────────────── HUD（顶部状态条） ────────────────────────────

def draw_top_hud(frame: np.ndarray,
                 action_id: int,
                 fps: float,
                 detection_count: int,
                 action_names: Dict[int, str] = None,
                 action_colors: Dict[int, Tuple[int, int, int]] = None,
                 height: int = 48) -> np.ndarray:
    """顶部半透明 HUD：显示当前 ACTION、检测数量、FPS。"""
    action_names  = action_names  or DEFAULT_ACTION_NAMES
    action_colors = action_colors or DEFAULT_ACTION_COLORS

    h, w = frame.shape[:2]
    _alpha_rect(frame, (0, 0), (w, height), COLOR_BLACK, alpha=0.55)

    name  = action_names.get(action_id, f"ACT{action_id}")
    color = action_colors.get(action_id, COLOR_WHITE)

    # 左：ACTION
    _put_text(frame, "ACTION", (10, 18),  COLOR_GRAY, scale=0.4)
    _put_text(frame, name,     (10, 40),  color,      scale=0.85, thickness=2)

    # 中：检测计数
    cnt_txt = f"DETECTED: {detection_count}"
    cnt_w   = cv2.getTextSize(cnt_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
    _put_text(frame, cnt_txt, ((w - cnt_w) // 2, 30),
              COLOR_WHITE, scale=0.6, thickness=1)

    # 右：FPS
    fps_txt = f"FPS {fps:.1f}"
    fps_w   = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
    fps_color = COLOR_GREEN if fps >= 15 else (COLOR_ORANGE if fps >= 8 else COLOR_RED)
    _put_text(frame, fps_txt, (w - fps_w - 10, 30),
              fps_color, scale=0.6, thickness=2)

    return frame


# ──────────────────────────── 右侧详情面板 ────────────────────────────

def draw_info_panel(frame: np.ndarray,
                    detections: List,
                    target_tag_id: Optional[int] = None,
                    panel_width: int = 220,
                    top_offset: int = 56) -> np.ndarray:
    """
    在画面右侧绘制详情面板，列出所有检测到的 tag 的关键信息：
        ID / TYPE / DIST / YAW / PITCH / CONF / corner 范围
    """
    h, w = frame.shape[:2]
    x0 = w - panel_width
    y0 = top_offset

    # 计算面板高度
    title_h    = 26
    item_h     = 56
    items_n    = max(1, len(detections))
    panel_h    = title_h + item_h * items_n + 10
    panel_h    = min(panel_h, h - top_offset - 80)   # 留出小地图位置

    _alpha_rect(frame, (x0, y0), (w - 6, y0 + panel_h),
                COLOR_DARK, alpha=0.7)
    cv2.rectangle(frame, (x0, y0), (w - 6, y0 + panel_h), COLOR_GRAY, 1)

    _put_text(frame, "DETECTIONS", (x0 + 8, y0 + 18),
              COLOR_WHITE, scale=0.55, thickness=2)

    if not detections:
        _put_text(frame, "(no tag in view)", (x0 + 8, y0 + 44),
                  COLOR_GRAY, scale=0.5)
        return frame

    cy = y0 + title_h + 6
    for idx, det in enumerate(detections):
        is_target = (target_tag_id is not None and det.tag_id == target_tag_id)
        if cy + item_h > y0 + panel_h:    # 面板放不下了
            _put_text(frame,
                      f"... +{len(detections) - idx} more",
                      (x0 + 8, cy + 12), COLOR_GRAY, scale=0.45)
            break

        color = _team_color(det.tag_id)
        marker = ">>" if is_target else "  "
        line1 = f"{marker} #{idx + 1}  ID={det.tag_id} {_team_name(det.tag_id)}"
        _put_text(frame, line1, (x0 + 8, cy + 12), color, scale=0.5,
                  thickness=2 if is_target else 1)

        line2 = f"   D={det.distance:.2f}m  Y={det.yaw_deg:+.1f}d"
        _put_text(frame, line2, (x0 + 8, cy + 28), COLOR_WHITE, scale=0.42)

        line3 = f"   P={det.pitch_deg:+.1f}d  C={det.confidence:.0f}"
        _put_text(frame, line3, (x0 + 8, cy + 44), COLOR_GRAY, scale=0.42)

        cy += item_h

    return frame


# ──────────────────────────── 右下角鸟瞰小地图 ────────────────────────────

def draw_minimap(frame: np.ndarray,
                 detections: List,
                 target_tag_id: Optional[int] = None,
                 our_team_id: int = 1,
                 enemy_team_id: int = 2,
                 size: int = 160,
                 max_range_m: float = 3.0) -> np.ndarray:
    """
    在右下角绘制俯视小地图：
        - 中心 = 小车位置
        - 朝上 = 小车正前方
        - 圆点 = 各个 tag（颜色按队伍区分）
        - 同心圆 = 1m / 2m / 3m 距离参考

    使用相机系坐标 (tx, tz)：
        tx > 0 = 右
        tz > 0 = 前
    """
    h, w = frame.shape[:2]
    margin = 12
    x1 = w - margin
    y1 = h - margin
    x0 = x1 - size
    y0 = y1 - size

    # 背景
    _alpha_rect(frame, (x0, y0), (x1, y1), COLOR_BLACK, alpha=0.65)
    cv2.rectangle(frame, (x0, y0), (x1, y1), COLOR_GRAY, 1)

    cx_pix = (x0 + x1) // 2
    cy_pix = (y0 + y1) // 2

    # 距离同心圆（每米一圈，最远 max_range_m）
    px_per_m = (size / 2 - 6) / max_range_m
    for ring_m in (1.0, 2.0, 3.0):
        if ring_m > max_range_m:
            break
        r_px = int(ring_m * px_per_m)
        cv2.circle(frame, (cx_pix, cy_pix), r_px, COLOR_DARK, 1, cv2.LINE_AA)
        _put_text(frame, f"{ring_m:.0f}m",
                  (cx_pix + 2, cy_pix - r_px + 2),
                  COLOR_GRAY, scale=0.35)

    # 视野扇区（FOV ~60°，仅作示意）
    fov_half = 30  # 度
    end_y = cy_pix - int((size / 2 - 6))
    end_x_l = cx_pix - int((size / 2 - 6) * np.tan(np.radians(fov_half)))
    end_x_r = cx_pix + int((size / 2 - 6) * np.tan(np.radians(fov_half)))
    cv2.line(frame, (cx_pix, cy_pix), (end_x_l, end_y), COLOR_DARK, 1, cv2.LINE_AA)
    cv2.line(frame, (cx_pix, cy_pix), (end_x_r, end_y), COLOR_DARK, 1, cv2.LINE_AA)

    # 小车（中心三角形）
    car = np.array([[cx_pix,     cy_pix - 8],
                    [cx_pix - 6, cy_pix + 6],
                    [cx_pix + 6, cy_pix + 6]], np.int32)
    cv2.fillPoly(frame, [car], COLOR_WHITE)

    # 画每个 tag
    for det in detections:
        # 相机系：t[0]=右, t[2]=前
        tx, tz = float(det.pose_t[0]), float(det.pose_t[2])
        # 限幅到地图范围
        tx_c = max(-max_range_m, min(max_range_m, tx))
        tz_c = max(-max_range_m, min(max_range_m, tz))
        px = int(cx_pix + tx_c * px_per_m)
        py = int(cy_pix - tz_c * px_per_m)
        is_target = (target_tag_id is not None and det.tag_id == target_tag_id)
        color = _team_color(det.tag_id)
        radius = 7 if is_target else 5
        cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), radius + 1, COLOR_WHITE, 1, cv2.LINE_AA)
        _put_text(frame, f"{det.tag_id}", (px + 7, py + 4),
                  COLOR_WHITE, scale=0.4, thickness=1)

    # 标题
    _put_text(frame, "TOP-DOWN VIEW", (x0 + 6, y0 + 14),
              COLOR_WHITE, scale=0.4)

    return frame


# ──────────────────────────── 底部状态条 ────────────────────────────

def draw_bottom_bar(frame: np.ndarray,
                    action_result,
                    height: int = 56) -> np.ndarray:
    """
    底部三行信息：
        Row 1 → TARGET：tag_id / type / dist / yaw / pitch
        Row 2 → CMD   ：v / omega / duration
        Row 3 → reason
    """
    h, w = frame.shape[:2]
    y0 = h - height
    _alpha_rect(frame, (0, y0), (w, h), COLOR_BLACK, alpha=0.55)

    a = action_result
    has_target = (a.tag_id != 0xFF and a.tag_id != 255)

    if has_target:
        tline = (f"TARGET  ID={a.tag_id} "
                 f"({_team_name(a.tag_id)})  "
                 f"D={a.dist_m:.2f}m  "
                 f"Y={a.yaw_deg:+.1f}d  "
                 f"P={a.pitch_deg:+.1f}d  "
                 f"t=({a.tx_m:+.2f},{a.ty_m:+.2f},{a.tz_m:+.2f})m")
        _put_text(frame, tline, (8, y0 + 14), COLOR_YELLOW, scale=0.45)
    else:
        _put_text(frame, "TARGET  none",
                  (8, y0 + 14), COLOR_GRAY, scale=0.45)

    cline = (f"CMD     v={a.v_pct:+.0f}%  "
             f"omega={a.omega_pct:+.0f}%  "
             f"dur={a.duration_ms}ms")
    _put_text(frame, cline, (8, y0 + 32), COLOR_CYAN, scale=0.45)

    rline = f"REASON  {a.reason}"
    if len(rline) > 100:
        rline = rline[:100] + "..."
    _put_text(frame, rline, (8, y0 + 50), COLOR_WHITE, scale=0.42)

    return frame


# ──────────────────────────── 一站式入口 ────────────────────────────

def render_overlay(frame: np.ndarray,
                   detections: List,
                   action_result,
                   camera_matrix: np.ndarray,
                   dist_coeffs: np.ndarray,
                   tag_size_m: float,
                   fps: float = 0.0,
                   our_team_id: int = 1,
                   enemy_team_id: int = 2,
                   action_names: Optional[Dict[int, str]] = None,
                   action_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
                   show_minimap: bool = True,
                   show_info_panel: bool = True) -> np.ndarray:
    """
    一站式渲染：把所有可视化元素叠加到 frame 上。
    上层调用者只需要传入"检测结果 + 决策结果"，无需关心绘制细节。
    """
    target_id = action_result.tag_id if (
        action_result and action_result.tag_id not in (0xFF, 255)
    ) else None

    # 1) tag 框 + 坐标轴 + 标签
    draw_all_detections(frame, detections,
                        camera_matrix, dist_coeffs, tag_size_m,
                        target_tag_id=target_id)

    # 2) 顶部 HUD
    draw_top_hud(frame,
                 action_id=action_result.action,
                 fps=fps,
                 detection_count=len(detections),
                 action_names=action_names,
                 action_colors=action_colors)

    # 3) 右侧信息面板
    if show_info_panel:
        draw_info_panel(frame, detections, target_tag_id=target_id)

    # 4) 右下角小地图
    if show_minimap:
        draw_minimap(frame, detections, target_tag_id=target_id,
                     our_team_id=our_team_id, enemy_team_id=enemy_team_id)

    # 5) 底部状态条
    draw_bottom_bar(frame, action_result)

    return frame
