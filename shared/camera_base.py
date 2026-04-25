"""
shared/camera_base.py
摄像头抽象接口（Camera HAL）

所有平台的摄像头实现都继承此基类，确保上层算法（AprilTagDetector、strategy）
不依赖任何具体硬件驱动。

实现列表：
  - vision_raspberrypi/src/camera_manager.py  → CameraManager(CameraBase)   picamera2
  - training/demo_pc/camera_opencv.py         → OpenCVCamera(CameraBase)    cv2.VideoCapture
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class CameraBase(ABC):
    """摄像头抽象基类。子类只需实现 open / read / close 三个方法。"""

    # 子类应在 __init__ 中设置以下属性
    width:  int = 640
    height: int = 480
    actual_fps: float = 0.0

    @abstractmethod
    def open(self) -> bool:
        """
        初始化并启动摄像头。
        Returns:
            True = 成功，False = 失败
        """

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧图像。
        Returns:
            (ok, frame_bgr)  —— frame 为 OpenCV BGR 格式 uint8 ndarray
            ok=False 时 frame 为 None
        """

    @abstractmethod
    def close(self) -> None:
        """释放摄像头资源。"""

    def is_open(self) -> bool:
        """默认实现，子类可覆写。"""
        return False

    # ── 上下文管理器支持（with 语句）──
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
