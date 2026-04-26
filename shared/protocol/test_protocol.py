"""
shared/protocol/test_protocol.py
协议双向自测脚本（纯 Python，不依赖 C 编译）。

测试内容：
1. CRC8 已知向量测试
2. 状态帧 pack / unpack 往返一致性
3. 控制帧 pack / unpack 往返一致性
4. 状态机解析器（FrameParser）流式解析
5. 边界值测试（int16 最大/最小、uint16 最大）
6. CRC 错误检测

运行：
    python test_protocol.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from protocol import (
    crc8, VisionData, CmdData, pack_vision, pack_cmd,
    unpack_vision, unpack_cmd, FrameParser, make_vision_frame, make_cmd_frame,
    TAG_ID_NONE, TAG_TYPE_NEUTRAL, TAG_TYPE_BLUE, TAG_TYPE_YELLOW,
    FLAG_VALID, FLAG_MULTI_TAG, ACTION_STOP, ACTION_VELOCITY, ACTION_PUSH,
    VISION_FRAME_LEN, CMD_FRAME_LEN, TYPE_VISION, TYPE_CMD,
)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
_failures = 0


def check(name: str, condition: bool):
    global _failures
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}")
        _failures += 1


# ──────────────────────────── 1. CRC8 ────────────────────────────

def test_crc8():
    print("== CRC8 向量测试 ==")
    # CRC-8/MAXIM 已知向量："123456789" -> 0xA1
    check("已知向量 '123456789' -> 0xA1",
          crc8(b"123456789") == 0xA1)
    check("空字节串 -> 0x00", crc8(b"") == 0x00)
    check("单字节 0x00 -> 0x00", crc8(b"\x00") == 0x00)


# ──────────────────────────── 2. 状态帧往返 ────────────────────────────

def test_vision_roundtrip():
    print("\n== 状态帧 pack/unpack 往返 ==")

    cases = [
        VisionData(seq=0, tag_id=0, tag_type=TAG_TYPE_NEUTRAL,
                   yaw_cdeg=0, pitch_cdeg=0, dist_mm=0,
                   tx_mm=0, ty_mm=0, tz_mm=0, flags=FLAG_VALID),
        VisionData(seq=255, tag_id=TAG_ID_NONE, tag_type=0xFF,
                   yaw_cdeg=32767, pitch_cdeg=-32768, dist_mm=65535,
                   tx_mm=32767, ty_mm=-32768, tz_mm=0, flags=0xFF),
        VisionData(seq=100, tag_id=1, tag_type=TAG_TYPE_BLUE,
                   yaw_cdeg=4500, pitch_cdeg=-1000, dist_mm=1200,
                   tx_mm=300, ty_mm=-150, tz_mm=1170, flags=FLAG_VALID),
    ]

    for i, d in enumerate(cases):
        frame = pack_vision(d)
        check(f"case{i} 帧长度 = {VISION_FRAME_LEN}", len(frame) == VISION_FRAME_LEN)
        check(f"case{i} 帧头 AA 55 01", frame[:3] == bytes([0xAA, 0x55, 0x01]))

        recovered = unpack_vision(frame)
        check(f"case{i} unpack 不为 None", recovered is not None)
        if recovered:
            check(f"case{i} seq 匹配",        recovered.seq       == d.seq)
            check(f"case{i} tag_id 匹配",     recovered.tag_id    == d.tag_id)
            check(f"case{i} yaw_cdeg 匹配",   recovered.yaw_cdeg  == d.yaw_cdeg)
            check(f"case{i} pitch_cdeg 匹配", recovered.pitch_cdeg == d.pitch_cdeg)
            check(f"case{i} dist_mm 匹配",    recovered.dist_mm   == d.dist_mm)
            check(f"case{i} tx_mm 匹配",      recovered.tx_mm     == d.tx_mm)
            check(f"case{i} ty_mm 匹配",      recovered.ty_mm     == d.ty_mm)
            check(f"case{i} tz_mm 匹配",      recovered.tz_mm     == d.tz_mm)
            check(f"case{i} flags 匹配",      recovered.flags     == d.flags)


# ──────────────────────────── 3. 控制帧往返 ────────────────────────────

def test_cmd_roundtrip():
    print("\n== 控制帧 pack/unpack 往返 ==")

    cases = [
        CmdData(seq=0, action=ACTION_STOP, v=0, omega=0, duration_ms=0),
        CmdData(seq=127, action=ACTION_VELOCITY, v=1000, omega=-1000,
                duration_ms=500),
        CmdData(seq=255, action=ACTION_PUSH, v=-500, omega=300,
                duration_ms=65535),
    ]

    for i, d in enumerate(cases):
        frame = pack_cmd(d)
        check(f"case{i} 帧长度 = {CMD_FRAME_LEN}", len(frame) == CMD_FRAME_LEN)
        check(f"case{i} 帧头 AA 55 02", frame[:3] == bytes([0xAA, 0x55, 0x02]))

        recovered = unpack_cmd(frame)
        check(f"case{i} unpack 不为 None", recovered is not None)
        if recovered:
            check(f"case{i} seq 匹配",         recovered.seq         == d.seq)
            check(f"case{i} action 匹配",       recovered.action      == d.action)
            check(f"case{i} v 匹配",            recovered.v           == d.v)
            check(f"case{i} omega 匹配",        recovered.omega       == d.omega)
            check(f"case{i} duration_ms 匹配",  recovered.duration_ms == d.duration_ms)


# ──────────────────────────── 4. 流式解析器 ────────────────────────────

def test_frame_parser():
    print("\n== FrameParser 流式解析 ==")

    parser = FrameParser()

    v = VisionData(seq=42, tag_id=1, tag_type=TAG_TYPE_BLUE,
                   yaw_cdeg=3000, pitch_cdeg=-500, dist_mm=800,
                   tx_mm=100, ty_mm=-50, tz_mm=795, flags=FLAG_VALID)
    vision_frame = pack_vision(v)

    c = CmdData(seq=43, action=ACTION_VELOCITY, v=500, omega=-200, duration_ms=0)
    cmd_frame = pack_cmd(c)

    # 两帧连在一起流式喂入
    stream = vision_frame + cmd_frame + b"\x00\x11\x22"  # 末尾夹杂垃圾字节

    results = []
    for b in stream:
        r = parser.feed(b)
        if r is not None:
            results.append(r)

    vision_results = [r for r in results if isinstance(r, VisionData)]
    cmd_results    = [r for r in results if isinstance(r, CmdData)]

    check("解析到 1 个状态帧", len(vision_results) == 1)
    check("解析到 1 个控制帧", len(cmd_results)    == 1)

    if vision_results:
        vr = vision_results[0]
        check("状态帧 seq=42",     vr.seq     == 42)
        check("状态帧 tag_id=1",   vr.tag_id  == 1)
        check("状态帧 yaw=3000",   vr.yaw_cdeg == 3000)

    if cmd_results:
        cr = cmd_results[0]
        check("控制帧 seq=43",     cr.seq    == 43)
        check("控制帧 v=500",      cr.v      == 500)
        check("控制帧 omega=-200", cr.omega  == -200)


# ──────────────────────────── 5. CRC 错误检测 ────────────────────────────

def test_crc_error():
    print("\n== CRC 错误检测 ==")

    frame = pack_vision(VisionData(seq=1, tag_id=0, tag_type=0,
                                   yaw_cdeg=100, pitch_cdeg=0, dist_mm=500,
                                   tx_mm=0, ty_mm=0, tz_mm=500, flags=FLAG_VALID))
    corrupted = bytearray(frame)
    corrupted[5] ^= 0xFF  # 翻转一个数据字节

    check("损坏帧 unpack_vision 返回 None", unpack_vision(bytes(corrupted)) is None)

    parser = FrameParser()
    results = []
    for b in corrupted:
        r = parser.feed(b)
        if r is not None:
            results.append(r)
    check("FrameParser 检测到 CRC 错误（返回 False）",
          any(r is False for r in results))


# ──────────────────────────── 6. 工厂函数 ────────────────────────────

def test_factory():
    print("\n== 工厂函数测试 ==")

    frame = make_vision_frame(
        seq=10, tag_id=1, tag_type=TAG_TYPE_BLUE,
        yaw_deg=45.0, pitch_deg=-10.0, dist_m=1.5,
        tx_m=0.3, ty_m=-0.1, tz_m=1.47,
    )
    result = unpack_vision(frame)
    check("工厂函数状态帧可解包", result is not None)
    if result:
        check("yaw_deg 45° -> yaw_cdeg 4500", result.yaw_cdeg == 4500)
        check("dist_m 1.5 -> dist_mm 1500",   result.dist_mm  == 1500)

    cmd_frame = make_cmd_frame(seq=11, action=ACTION_PUSH,
                               v_pct=60.0, omega_pct=-30.0, duration_ms=200)
    cmd_result = unpack_cmd(cmd_frame)
    check("工厂函数控制帧可解包", cmd_result is not None)
    if cmd_result:
        check("v_pct 60% -> v 600",       cmd_result.v     == 600)
        check("omega_pct -30% -> omega -300", cmd_result.omega == -300)


# ──────────────────────────── 主入口 ────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  协议双向自测")
    print("=" * 50)

    test_crc8()
    test_vision_roundtrip()
    test_cmd_roundtrip()
    test_frame_parser()
    test_crc_error()
    test_factory()

    print("\n" + "=" * 50)
    if _failures == 0:
        print(f"\033[92m所有测试通过！\033[0m")
    else:
        print(f"\033[91m{_failures} 个测试失败！\033[0m")
        sys.exit(1)
