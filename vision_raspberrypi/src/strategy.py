"""
vision_raspberrypi/src/strategy.py
决策引擎：根据 AprilTag 检测结果生成控制指令。
"""

import math
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "shared"))

from protocol.protocol import (
    ACTION_STOP, ACTION_VELOCITY, ACTION_PUSH,
    ACTION_AVOID, ACTION_PROTECT, ACTION_SEARCH,
    TAG_TYPE_NEUTRAL, TAG_TYPE_BLUE, TAG_TYPE_YELLOW,
    FLAG_VALID, FLAG_MULTI_TAG,
)
from .apriltag_detector import DetectionResult


class BlockType(IntEnum):
    OUR_TEAM   = 1
    ENEMY_TEAM = 2
    NEUTRAL    = 0
    UNKNOWN    = 255


@dataclass
class ActionResult:
    """单步决策输出"""
    action:      int    # PROTO_ACTION_*
    v_pct:       float  # 前进速度 -100~+100 %
    omega_pct:   float  # 角速度   -100~+100 %（正=左转）
    duration_ms: int    # 0=持续
    reason:      str    # 调试用文字说明
    # 原始视觉信息（用于状态帧）
    tag_id:      int    = 255
    tag_type:    int    = 255
    yaw_deg:     float  = 0.0
    pitch_deg:   float  = 0.0
    dist_m:      float  = 0.0
    tx_m:        float  = 0.0
    ty_m:        float  = 0.0
    tz_m:        float  = 0.0
    flags:       int    = 0


class Strategy:
    """简单规则型决策引擎"""

    def __init__(self, config: dict):
        """
        Args:
            config: 来自 config.yaml 的 strategy 节点
        """
        self._log = logging.getLogger(__name__)
        cfg = config

        # 队伍配置（比赛前在 config.yaml 里改）
        self.our_id    = cfg.get("our_team_id",  1)
        self.enemy_id  = cfg.get("enemy_team_id", 2)
        self.neutral_id = 0

        # 距离阈值（米）
        d = cfg.get("distances", {})
        self.push_dist    = d.get("push_dist",    0.30)
        self.approach_dist= d.get("approach_dist", 1.20)
        self.avoid_dist   = d.get("avoid_dist",    0.40)

        # 速度参数（百分比）
        s = cfg.get("speeds", {})
        self.approach_v  = s.get("approach",  40.0)
        self.push_v      = s.get("push",      60.0)
        self.turn_omega  = s.get("turn",      35.0)
        self.search_omega= s.get("search",   -25.0)  # 负=左转搜索

    # ────────────────── 主接口 ──────────────────

    def decide(self, detections: List[DetectionResult]) -> ActionResult:
        """
        根据本帧检测结果决策。
        Returns:
            ActionResult（包含控制指令 + 视觉信息）
        """
        if not detections:
            return self._search_action()

        # 按优先级选目标：敌方 > 中立（避障）> 我方
        enemy   = self._pick_by_type(detections, self.enemy_id)
        neutral = self._pick_by_type(detections, self.neutral_id)
        ours    = self._pick_by_type(detections, self.our_id)

        primary = enemy or neutral or ours or detections[0]

        # 填充视觉信息
        tag_type = self._classify(primary.tag_id)
        flags    = FLAG_VALID
        if len(detections) > 1:
            flags |= FLAG_MULTI_TAG

        base = dict(
            tag_id    = primary.tag_id,
            tag_type  = int(tag_type),
            yaw_deg   = primary.yaw_deg,
            pitch_deg = primary.pitch_deg,
            dist_m    = primary.distance,
            tx_m      = float(primary.pose_t[0]),
            ty_m      = float(primary.pose_t[1]),
            tz_m      = float(primary.pose_t[2]),
            flags     = flags,
        )

        if primary.tag_id == self.enemy_id:
            return self._attack_action(primary, **base)
        elif primary.tag_id == self.neutral_id:
            return self._avoid_action(primary, **base)
        else:
            return self._protect_action(primary, **base)

    # ────────────────── 具体决策 ──────────────────

    def _attack_action(self, det: DetectionResult, **kw) -> ActionResult:
        """朝敌方能量块推进"""
        yaw   = det.yaw_deg
        dist  = det.distance

        # P 控制朝向
        omega = self._yaw_to_omega(yaw)

        if dist <= self.push_dist:
            return ActionResult(action=ACTION_PUSH,
                                v_pct=self.push_v, omega_pct=omega,
                                duration_ms=0,
                                reason=f"推进中 dist={dist:.2f}m yaw={yaw:.1f}°",
                                **kw)
        else:
            v = self.approach_v if abs(yaw) < 30 else 0.0
            return ActionResult(action=ACTION_VELOCITY,
                                v_pct=v, omega_pct=omega,
                                duration_ms=0,
                                reason=f"接近敌方 dist={dist:.2f}m yaw={yaw:.1f}°",
                                **kw)

    def _avoid_action(self, det: DetectionResult, **kw) -> ActionResult:
        """绕开中立能量块"""
        if det.distance > self.avoid_dist:
            return ActionResult(action=ACTION_VELOCITY,
                                v_pct=self.approach_v, omega_pct=0.0,
                                duration_ms=0, reason="中立块距离安全，继续前进",
                                **kw)
        omega = self.turn_omega if det.yaw_deg < 0 else -self.turn_omega
        return ActionResult(action=ACTION_AVOID,
                            v_pct=0.0, omega_pct=omega,
                            duration_ms=300, reason=f"避让中立块 dist={det.distance:.2f}m",
                            **kw)

    def _protect_action(self, det: DetectionResult, **kw) -> ActionResult:
        """守护我方能量块"""
        omega = self._yaw_to_omega(det.yaw_deg)
        return ActionResult(action=ACTION_PROTECT,
                            v_pct=0.0, omega_pct=omega,
                            duration_ms=0,
                            reason=f"监视我方块 dist={det.distance:.2f}m",
                            **kw)

    def _search_action(self) -> ActionResult:
        """原地旋转搜索"""
        return ActionResult(action=ACTION_SEARCH,
                            v_pct=0.0, omega_pct=self.search_omega,
                            duration_ms=0, reason="未检测到目标，搜索中",
                            tag_id=0xFF, tag_type=0xFF,
                            yaw_deg=0.0, pitch_deg=0.0,
                            dist_m=0.0, tx_m=0.0, ty_m=0.0, tz_m=0.0,
                            flags=0)

    # ────────────────── 工具 ──────────────────

    def _pick_by_type(self, dets: List[DetectionResult],
                      tag_id: int) -> Optional[DetectionResult]:
        candidates = [d for d in dets if d.tag_id == tag_id]
        return min(candidates, key=lambda d: d.distance) if candidates else None

    def _classify(self, tag_id: int) -> BlockType:
        if tag_id == self.our_id:
            return BlockType.OUR_TEAM
        if tag_id == self.enemy_id:
            return BlockType.ENEMY_TEAM
        if tag_id == self.neutral_id:
            return BlockType.NEUTRAL
        return BlockType.UNKNOWN

    def _yaw_to_omega(self, yaw_deg: float) -> float:
        """偏航角转角速度指令（简单 P 控制，右正左负）"""
        gain = self.turn_omega / 30.0    # 30° 对应满转速
        omega = -yaw_deg * gain          # 负号：目标在右(yaw>0)→右转(omega<0)
        return max(-self.turn_omega, min(self.turn_omega, omega))
