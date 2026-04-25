"""
vision_raspberrypi/src/utils.py
公共工具函数。
"""

import time
import logging


class LoopRateController:
    """
    固定频率控制器（不使用 time.sleep 精确版）。

    用法::

        ctrl = LoopRateController(20)  # 20 Hz
        while True:
            do_work()
            ctrl.sleep()
    """

    def __init__(self, hz: float):
        self._period = 1.0 / hz
        self._last   = time.monotonic()

    def sleep(self):
        """等待到下一个时间槽"""
        elapsed  = time.monotonic() - self._last
        remaining = self._period - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()

    @property
    def period(self) -> float:
        return self._period


class FirstOrderFilter:
    """一阶低通滤波器"""

    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: 平滑系数，越小越平滑（0~1）
        """
        self._alpha = alpha
        self._val   = None

    def update(self, new_val: float) -> float:
        if self._val is None:
            self._val = new_val
        else:
            self._val = self._alpha * new_val + (1 - self._alpha) * self._val
        return self._val

    def reset(self):
        self._val = None

    @property
    def value(self):
        return self._val


def setup_logging(level: str = "INFO"):
    """配置全局日志格式"""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level   = numeric,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%H:%M:%S",
    )
