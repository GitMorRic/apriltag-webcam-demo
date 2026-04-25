"""
vision_raspberrypi/src/apriltag_detector.py
AprilTag 检测与位姿估计。
"""

import cv2
import numpy as np
import math
import logging
import yaml
from dataclasses import dataclass
from typing import List, Optional, Tuple

# 兼容三种 apriltag 包：
#   - apriltag（PyPI 原版 / conda-forge / Linux 可直接 pip install）
#   - pupil_apriltags（Windows 推荐，pip install pupil-apriltags）
# 两者 API 略有差异，统一在 AprilTagDetector 内部处理。
try:
    import apriltag
    _APRILTAG_BACKEND = "apriltag"
except ImportError:
    try:
        import pupil_apriltags as apriltag
        _APRILTAG_BACKEND = "pupil_apriltags"
    except ImportError as e:
        raise ImportError(
            "未找到 apriltag 库，请安装其中之一：\n"
            "  pip install pupil-apriltags        （Windows 推荐）\n"
            "  pip install apriltag               （Linux / macOS）\n"
            "  conda install -c conda-forge apriltag\n"
            f"原始错误: {e}"
        )


@dataclass
class DetectionResult:
    """单个 AprilTag 的检测结果"""
    tag_id:    int
    center:    Tuple[float, float]   # 图像中心（像素）
    corners:   np.ndarray            # (4,2) 四角点（像素）
    pose_t:    np.ndarray            # (3,) 平移向量（相机坐标系，米）
    pose_R:    np.ndarray            # (3,3) 旋转矩阵
    distance:  float                 # 到相机的直线距离（米）
    yaw_deg:   float                 # 水平偏航角（度，右正左负）
    pitch_deg: float                 # 俯仰角（度）
    confidence: float                # 检测置信度（decision_margin）


class AprilTagDetector:
    """AprilTag 检测器（Tag36h11，基于 python-apriltag）"""

    def __init__(self, calib_cfg: dict, detect_cfg: dict):
        """
        Args:
            calib_cfg:  来自 config.yaml 的 calibration 节点
            detect_cfg: 来自 config.yaml 的 apriltag 节点
        """
        self._log = logging.getLogger(__name__)

        # 相机内参
        fx = calib_cfg["fx"]
        fy = calib_cfg["fy"]
        cx = calib_cfg["cx"]
        cy = calib_cfg["cy"]
        self._camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                                        dtype=np.float64)
        dist = calib_cfg.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])
        self._dist_coeffs = np.array(dist, dtype=np.float64)

        self._tag_size = detect_cfg.get("tag_size", 0.12)   # 米

        # 初始化检测器（兼容两种 API）
        # - conda-forge / pip apriltag：使用 DetectorOptions 对象
        # - pupil-apriltags：直接把参数传给 Detector 构造函数
        det_kwargs = dict(
            families      = detect_cfg.get("family", "tag36h11"),
            nthreads      = int(detect_cfg.get("nthreads", 4)),
            quad_decimate = float(detect_cfg.get("quad_decimate", 2.0)),
            quad_sigma    = float(detect_cfg.get("quad_sigma", 0.0)),
            refine_edges  = int(detect_cfg.get("refine_edges", True)),
        )
        try:
            opts = apriltag.DetectorOptions(**det_kwargs)   # conda-forge API
            self._detector = apriltag.Detector(opts)
        except AttributeError:
            self._detector = apriltag.Detector(**det_kwargs)  # pupil-apriltags API
        self._log.info("AprilTag 检测器初始化完成（tag_size=%.3fm）", self._tag_size)

    # ────────────────── 主检测接口 ──────────────────

    def detect(self, image_bgr: np.ndarray) -> List[DetectionResult]:
        """
        检测图像中的所有 AprilTag。
        Args:
            image_bgr: OpenCV BGR 图像
        Returns:
            检测结果列表（按距离从近到远排序）
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        detections = self._detector.detect(gray)

        results = []
        for det in detections:
            try:
                result = self._process_detection(det)
                if result:
                    results.append(result)
            except Exception as e:
                self._log.warning("处理 tag%d 失败: %s", det.tag_id, e)

        results.sort(key=lambda r: r.distance)
        return results

    # ────────────────── 内部处理 ──────────────────

    def _process_detection(self, det) -> Optional[DetectionResult]:
        half = self._tag_size / 2.0
        # AprilTag 角点顺序：左下、右下、右上、左上
        obj_pts = np.array([
            [-half, -half, 0.0],
            [ half, -half, 0.0],
            [ half,  half, 0.0],
            [-half,  half, 0.0],
        ], dtype=np.float32)

        img_pts = det.corners.astype(np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts,
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        t    = tvec.flatten()

        # 水平距离（投影到地面平面）
        dist_horiz = math.sqrt(t[0]**2 + t[2]**2)
        # 偏航角：相机系 X/Z → 水平面角度，右正左负
        yaw_deg   = math.degrees(math.atan2(t[0], t[2]))
        # 俯仰角：Y 轴（向下为正）
        pitch_deg = math.degrees(math.atan2(-t[1], t[2]))

        return DetectionResult(
            tag_id    = det.tag_id,
            center    = tuple(det.center),
            corners   = det.corners.copy(),
            pose_t    = t,
            pose_R    = R,
            distance  = float(np.linalg.norm(t)),
            yaw_deg   = yaw_deg,
            pitch_deg = pitch_deg,
            confidence = float(det.decision_margin),
        )

    # ────────────────── 可视化（薄包装） ──────────────────
    # 实际绘制实现统一放在 shared/visualization.py，
    # 这样 PC demo / 树莓派调试 / 单元测试可以共用同一套 UI。

    def draw_detections(self, image_bgr: np.ndarray,
                        results: List[DetectionResult],
                        target_tag_id: Optional[int] = None) -> np.ndarray:
        """
        在图像上绘制所有检测结果（边框 + 角点 + 3D 坐标轴 + 标签）。
        若指定 target_tag_id，对应 tag 会被高亮。
        """
        # 延迟导入避免循环依赖；shared 路径已由调用方加入 sys.path
        try:
            from shared.visualization import draw_all_detections
        except ImportError:
            # 兼容直接从 vision_raspberrypi/ 启动、未把仓库根加入 sys.path 的情况
            import sys, os
            _root = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "..", ".."))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from shared.visualization import draw_all_detections

        out = image_bgr.copy()
        return draw_all_detections(
            out, results,
            camera_matrix=self._camera_matrix,
            dist_coeffs=self._dist_coeffs,
            tag_size_m=self._tag_size,
            target_tag_id=target_tag_id,
        )
