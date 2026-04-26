"""
shared/protocol/protocol.py
与 protocol.h/c 完全等价的 Python 实现，供树莓派上位机使用。

帧格式见 protocol.h 文件头注释。
"""

import struct
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ──────────────────────────── 协议常量 ────────────────────────────

HEADER_0 = 0xAA
HEADER_1 = 0x55

TYPE_VISION = 0x01
TYPE_CMD    = 0x02

VISION_PAYLOAD_LEN = 16   # SEQ(1)+TAG_ID(1)+TAG_TYPE(1)+YAW(2)+PITCH(2)+DIST(2)+TX(2)+TY(2)+TZ(2)+FLAGS(1)
CMD_PAYLOAD_LEN    = 10   # SEQ(1)+ACTION(1)+V(2)+OMEGA(2)+DUR(2)+RES(2)

VISION_FRAME_LEN = 21    # 2+1+1+16+1
CMD_FRAME_LEN    = 15    # 2+1+1+10+1

TAG_ID_NONE       = 0xFF

TAG_TYPE_NEUTRAL  = 0x00
TAG_TYPE_BLUE     = 0x01
TAG_TYPE_YELLOW   = 0x02
TAG_TYPE_UNKNOWN  = 0xFF

FLAG_VALID      = 1 << 0
FLAG_MULTI_TAG  = 1 << 1
FLAG_LOW_CONF   = 1 << 2

ACTION_STOP     = 0x00
ACTION_VELOCITY = 0x01
ACTION_PUSH     = 0x02
ACTION_AVOID    = 0x03
ACTION_PROTECT  = 0x04
ACTION_SEARCH   = 0x05

# ──────────────────────────── CRC-8/MAXIM ────────────────────────────

_CRC8_TABLE = [
    0x00, 0x5E, 0xBC, 0xE2, 0x61, 0x3F, 0xDD, 0x83,
    0xC2, 0x9C, 0x7E, 0x20, 0xA3, 0xFD, 0x1F, 0x41,
    0x9D, 0xC3, 0x21, 0x7F, 0xFC, 0xA2, 0x40, 0x1E,
    0x5F, 0x01, 0xE3, 0xBD, 0x3E, 0x60, 0x82, 0xDC,
    0x23, 0x7D, 0x9F, 0xC1, 0x42, 0x1C, 0xFE, 0xA0,
    0xE1, 0xBF, 0x5D, 0x03, 0x80, 0xDE, 0x3C, 0x62,
    0xBE, 0xE0, 0x02, 0x5C, 0xDF, 0x81, 0x63, 0x3D,
    0x7C, 0x22, 0xC0, 0x9E, 0x1D, 0x43, 0xA1, 0xFF,
    0x46, 0x18, 0xFA, 0xA4, 0x27, 0x79, 0x9B, 0xC5,
    0x84, 0xDA, 0x38, 0x66, 0xE5, 0xBB, 0x59, 0x07,
    0xDB, 0x85, 0x67, 0x39, 0xBA, 0xE4, 0x06, 0x58,
    0x19, 0x47, 0xA5, 0xFB, 0x78, 0x26, 0xC4, 0x9A,
    0x65, 0x3B, 0xD9, 0x87, 0x04, 0x5A, 0xB8, 0xE6,
    0xA7, 0xF9, 0x1B, 0x45, 0xC6, 0x98, 0x7A, 0x24,
    0xF8, 0xA6, 0x44, 0x1A, 0x99, 0xC7, 0x25, 0x7B,
    0x3A, 0x64, 0x86, 0xD8, 0x5B, 0x05, 0xE7, 0xB9,
    0x8C, 0xD2, 0x30, 0x6E, 0xED, 0xB3, 0x51, 0x0F,
    0x4E, 0x10, 0xF2, 0xAC, 0x2F, 0x71, 0x93, 0xCD,
    0x11, 0x4F, 0xAD, 0xF3, 0x70, 0x2E, 0xCC, 0x92,
    0xD3, 0x8D, 0x6F, 0x31, 0xB2, 0xEC, 0x0E, 0x50,
    0xAF, 0xF1, 0x13, 0x4D, 0xCE, 0x90, 0x72, 0x2C,
    0x6D, 0x33, 0xD1, 0x8F, 0x0C, 0x52, 0xB0, 0xEE,
    0x32, 0x6C, 0x8E, 0xD0, 0x53, 0x0D, 0xEF, 0xB1,
    0xF0, 0xAE, 0x4C, 0x12, 0x91, 0xCF, 0x2D, 0x73,
    0xCA, 0x94, 0x76, 0x28, 0xAB, 0xF5, 0x17, 0x49,
    0x08, 0x56, 0xB4, 0xEA, 0x69, 0x37, 0xD5, 0x8B,
    0x57, 0x09, 0xEB, 0xB5, 0x36, 0x68, 0x8A, 0xD4,
    0x95, 0xCB, 0x29, 0x77, 0xF4, 0xAA, 0x48, 0x16,
    0xE9, 0xB7, 0x55, 0x0B, 0x88, 0xD6, 0x34, 0x6A,
    0x2B, 0x75, 0x97, 0xC9, 0x4A, 0x14, 0xF6, 0xA8,
    0x74, 0x2A, 0xC8, 0x96, 0x15, 0x4B, 0xA9, 0xF7,
    0xB6, 0xE8, 0x0A, 0x54, 0xD7, 0x89, 0x6B, 0x35,
]


def crc8(data: bytes) -> int:
    """CRC-8/MAXIM，多项式 0x31，初值 0x00"""
    val = 0
    for b in data:
        val = _CRC8_TABLE[val ^ b]
    return val


# ──────────────────────────── 数据类 ────────────────────────────

@dataclass
class VisionData:
    """状态帧数据。所有角度单位 0.01°，距离/位置单位 mm。"""
    seq: int       = 0
    tag_id: int    = TAG_ID_NONE
    tag_type: int  = TAG_TYPE_UNKNOWN
    yaw_cdeg: int  = 0     # int16，水平偏航角，正值 = 目标在右
    pitch_cdeg: int = 0    # int16，俯仰角
    dist_mm: int   = 0     # uint16，水平投影距离
    tx_mm: int     = 0     # int16，相机系 X
    ty_mm: int     = 0     # int16，相机系 Y
    tz_mm: int     = 0     # int16，相机系 Z（正前方）
    flags: int     = 0


@dataclass
class CmdData:
    """控制帧数据。V/omega 范围 -1000~+1000（0.1% 单位）。"""
    seq: int          = 0
    action: int       = ACTION_STOP
    v: int            = 0      # int16
    omega: int        = 0      # int16
    duration_ms: int  = 0      # uint16，0=持续
    reserved: int     = 0      # uint16


# ──────────────────────────── 打包 ────────────────────────────

# 状态帧载荷打包格式：SEQ TAG_ID TAG_TYPE YAW PITCH DIST TX TY TZ FLAGS
# <   = 小端序
# BBB = 3× uint8
# hh  = 2× int16
# H   = 1× uint16
# hhh = 3× int16
# B   = uint8（flags）
_VISION_PAYLOAD_FMT = "<BBBhhHhhhB"
# 控制帧载荷：SEQ ACTION V OMEGA DUR RES
_CMD_PAYLOAD_FMT    = "<BBhhHH"


def pack_vision(data: VisionData) -> bytes:
    """打包状态帧，返回 VISION_FRAME_LEN 字节的 bytes。"""
    payload = struct.pack(
        _VISION_PAYLOAD_FMT,
        data.seq & 0xFF,
        data.tag_id & 0xFF,
        data.tag_type & 0xFF,
        _clamp_i16(data.yaw_cdeg),
        _clamp_i16(data.pitch_cdeg),
        _clamp_u16(data.dist_mm),
        _clamp_i16(data.tx_mm),
        _clamp_i16(data.ty_mm),
        _clamp_i16(data.tz_mm),
        data.flags & 0xFF,
    )
    assert len(payload) == VISION_PAYLOAD_LEN

    # CRC 计算范围：TYPE + LEN + payload（共 2 + 16 = 18 字节）
    crc_data = bytes([TYPE_VISION, VISION_PAYLOAD_LEN]) + payload
    checksum = crc8(crc_data)

    return bytes([HEADER_0, HEADER_1, TYPE_VISION, VISION_PAYLOAD_LEN]) + payload + bytes([checksum])


def pack_cmd(data: CmdData) -> bytes:
    """打包控制帧，返回 CMD_FRAME_LEN 字节的 bytes。"""
    payload = struct.pack(
        _CMD_PAYLOAD_FMT,
        data.seq & 0xFF,
        data.action & 0xFF,
        _clamp_i16(data.v),
        _clamp_i16(data.omega),
        _clamp_u16(data.duration_ms),
        _clamp_u16(data.reserved),
    )
    assert len(payload) == CMD_PAYLOAD_LEN

    crc_data = bytes([TYPE_CMD, CMD_PAYLOAD_LEN]) + payload
    checksum = crc8(crc_data)

    return bytes([HEADER_0, HEADER_1, TYPE_CMD, CMD_PAYLOAD_LEN]) + payload + bytes([checksum])


# ──────────────────────────── 解包 ────────────────────────────

def unpack_vision(frame: bytes) -> Optional[VisionData]:
    """
    解包一帧状态帧字节（需已确认帧头和 TYPE 正确）。
    返回 VisionData 或 None（CRC 错误）。
    """
    if len(frame) != VISION_FRAME_LEN:
        return None
    crc_data = frame[2:-1]
    if crc8(crc_data) != frame[-1]:
        return None
    pl = frame[4:-1]
    fields = struct.unpack(_VISION_PAYLOAD_FMT, pl)
    return VisionData(
        seq=fields[0], tag_id=fields[1], tag_type=fields[2],
        yaw_cdeg=fields[3], pitch_cdeg=fields[4], dist_mm=fields[5],
        tx_mm=fields[6], ty_mm=fields[7], tz_mm=fields[8], flags=fields[9],
    )


def unpack_cmd(frame: bytes) -> Optional[CmdData]:
    """解包一帧控制帧字节，返回 CmdData 或 None（CRC 错误）。"""
    if len(frame) != CMD_FRAME_LEN:
        return None
    crc_data = frame[2:-1]
    if crc8(crc_data) != frame[-1]:
        return None
    pl = frame[4:-1]
    fields = struct.unpack(_CMD_PAYLOAD_FMT, pl)
    return CmdData(
        seq=fields[0], action=fields[1], v=fields[2],
        omega=fields[3], duration_ms=fields[4], reserved=fields[5],
    )


# ──────────────────────────── 流式解析器 ────────────────────────────

class FrameParser:
    """
    字节流状态机解析器，逐字节喂入，自动识别完整帧。

    用法：
        parser = FrameParser()
        for byte in serial_stream:
            result = parser.feed(byte)
            if isinstance(result, VisionData):
                handle_vision(result)
            elif isinstance(result, CmdData):
                handle_cmd(result)
            elif result is False:
                handle_crc_error()
    """

    _S_HDR0    = 0
    _S_HDR1    = 1
    _S_TYPE    = 2
    _S_LEN     = 3
    _S_PAYLOAD = 4
    _S_CRC     = 5

    def __init__(self):
        self.reset()

    def reset(self):
        self._state    = self._S_HDR0
        self._buf      = bytearray()
        self._type     = 0
        self._len      = 0

    def feed(self, byte: int):
        """
        喂入一个字节（0~255）。
        返回：VisionData | CmdData（成功）| False（CRC 错误）| None（继续等待）
        """
        s = self._state

        if s == self._S_HDR0:
            if byte == HEADER_0:
                self._buf = bytearray([byte])
                self._state = self._S_HDR1

        elif s == self._S_HDR1:
            if byte == HEADER_1:
                self._buf.append(byte)
                self._state = self._S_TYPE
            else:
                self._state = self._S_HDR0

        elif s == self._S_TYPE:
            if byte in (TYPE_VISION, TYPE_CMD):
                self._type = byte
                self._buf.append(byte)
                self._state = self._S_LEN
            else:
                self._state = self._S_HDR0

        elif s == self._S_LEN:
            expected = VISION_PAYLOAD_LEN if self._type == TYPE_VISION else CMD_PAYLOAD_LEN
            if byte != expected:
                self._state = self._S_HDR0
                return None
            self._len = byte
            self._buf.append(byte)
            self._state = self._S_PAYLOAD

        elif s == self._S_PAYLOAD:
            self._buf.append(byte)
            if len(self._buf) == 4 + self._len:
                self._state = self._S_CRC

        elif s == self._S_CRC:
            self._buf.append(byte)
            self._state = self._S_HDR0
            frame = bytes(self._buf)
            if self._type == TYPE_VISION:
                result = unpack_vision(frame)
            else:
                result = unpack_cmd(frame)
            return result if result is not None else False

        return None


# ──────────────────────────── 工厂函数（便于上位机调用） ────────────────────────────

def make_vision_frame(
    seq: int,
    tag_id: int,
    tag_type: int,
    yaw_deg: float,
    pitch_deg: float,
    dist_m: float,
    tx_m: float,
    ty_m: float,
    tz_m: float,
    flags: int = FLAG_VALID,
) -> bytes:
    """
    工厂函数：接受「物理单位」参数，自动转换并打包状态帧。
    yaw/pitch 单位：度（float），dist/tx/ty/tz 单位：米（float）。
    """
    d = VisionData(
        seq        = seq,
        tag_id     = tag_id,
        tag_type   = tag_type,
        yaw_cdeg   = int(round(yaw_deg   * 100)),
        pitch_cdeg = int(round(pitch_deg * 100)),
        dist_mm    = int(round(dist_m    * 1000)),
        tx_mm      = int(round(tx_m      * 1000)),
        ty_mm      = int(round(ty_m      * 1000)),
        tz_mm      = int(round(tz_m      * 1000)),
        flags      = flags,
    )
    return pack_vision(d)


def make_cmd_frame(
    seq: int,
    action: int,
    v_pct: float,
    omega_pct: float,
    duration_ms: int = 0,
) -> bytes:
    """
    工厂函数：v_pct/omega_pct 为百分比（-100.0~+100.0），自动转 int16。
    """
    d = CmdData(
        seq         = seq,
        action      = action,
        v           = int(round(v_pct * 10)),
        omega       = int(round(omega_pct * 10)),
        duration_ms = duration_ms,
    )
    return pack_cmd(d)


# ──────────────────────────── 内部辅助 ────────────────────────────

def _clamp_i16(v: int) -> int:
    return max(-32768, min(32767, int(v)))

def _clamp_u16(v: int) -> int:
    return max(0, min(65535, int(v)))
