"""
training/demo_pc/camera_opencv.py
OpenCV 摄像头实现（适用于 PC 笔记本摄像头或 USB 摄像头）。

与 vision_raspberrypi/src/camera_manager.py 实现相同的 CameraBase 接口，
上层算法（AprilTagDetector / StrategyEngine）无需修改即可切换运行平台。
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))
from camera_base import CameraBase


class OpenCVCamera(CameraBase):
    """
    基于 cv2.VideoCapture 的摄像头，支持：
      - 笔记本内置摄像头（device_id=0）
      - USB 外接摄像头（device_id=1, 2, ...）
      - 本地视频文件（device_id="path/to/video.mp4"）
    """

    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480):
        self._log = logging.getLogger(__name__)
        self._device_id = device_id
        self.width  = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._open_flag = False
        self.actual_fps = 0.0
        self._frame_count = 0
        self._fps_ts = time.monotonic()

    # ────────────────── CameraBase 接口实现 ──────────────────

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            self._log.error("无法打开摄像头 device_id=%s", self._device_id)
            return False

        # 请求分辨率（不保证一定生效，取决于摄像头支持）
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # 读回实际分辨率
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._open_flag = True
        self._log.info("OpenCV 摄像头已启动：device_id=%s, 分辨率=%dx%d",
                       self._device_id, self.width, self.height)
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._open_flag or self._cap is None:
            return False, None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        self._update_fps()
        return True, frame   # OpenCV 默认已是 BGR

    def close(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None
        self._open_flag = False

    def is_open(self) -> bool:
        return self._open_flag

    # ────────────────── 内部 ──────────────────

    def _update_fps(self):
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            now = time.monotonic()
            dt = now - self._fps_ts
            if dt > 0:
                self.actual_fps = 30.0 / dt
            self._fps_ts = now
