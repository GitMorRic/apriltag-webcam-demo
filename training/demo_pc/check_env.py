"""
training/demo_pc/check_env.py
环境自检脚本 —— 逐项检查 demo_pc 运行所需的全部依赖。

运行方式：
    python training/demo_pc/check_env.py
"""

import sys
import os

PASS  = "[  OK  ]"
FAIL  = "[ FAIL ]"
WARN  = "[ WARN ]"
INFO  = "[ INFO ]"
SEP   = "─" * 56


def check(label: str, fn):
    try:
        result = fn()
        if result is True or result is None:
            print(f"{PASS}  {label}")
            return True
        elif isinstance(result, str):
            print(f"{PASS}  {label}  →  {result}")
            return True
        else:
            print(f"{FAIL}  {label}  →  {result}")
            return False
    except Exception as e:
        print(f"{FAIL}  {label}  →  {e}")
        return False


def warn(label: str, fn):
    try:
        result = fn()
        if result is True or result is None:
            print(f"{PASS}  {label}")
        elif isinstance(result, str):
            print(f"{PASS}  {label}  →  {result}")
        else:
            print(f"{WARN}  {label}  →  {result}")
    except Exception as e:
        print(f"{WARN}  {label}  →  {e}")


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ──────────────────────────────────────────────────────────
section("1. Python 版本")
check("Python >= 3.10",
      lambda: sys.version if sys.version_info >= (3, 10)
              else f"需要 3.10+，当前 {sys.version}")

section("2. 核心依赖")
check("numpy",  lambda: __import__("numpy").__version__)
check("cv2 (opencv-python)",
      lambda: __import__("cv2").__version__)
def _check_apriltag():
    # 注意：pip 安装的 pupil-apriltags 包的 import 名是 pupil_apriltags
    # 而 apriltag / conda-forge apriltag 的 import 名是 apriltag
    # 两者 API 不同，下面分别处理。
    backend = None
    mod = None
    try:
        import apriltag as mod
        backend = "apriltag"
    except ImportError:
        try:
            import pupil_apriltags as mod
            backend = "pupil_apriltags"
        except ImportError:
            return ("未找到 apriltag 库。请执行："
                    "pip install pupil-apriltags")

    ver = getattr(mod, "__version__", "无版本号")

    # apriltag 包是 DetectorOptions 风格，pupil_apriltags 是直接 kwargs 风格
    try:
        opts = mod.DetectorOptions(families="tag36h11", nthreads=1)
        _ = mod.Detector(opts)
        return f"backend={backend}（DetectorOptions API）  ver={ver}"
    except AttributeError:
        try:
            _ = mod.Detector(families="tag36h11", nthreads=1)
            return f"backend={backend}（kwargs API）  ver={ver}"
        except Exception as e2:
            raise RuntimeError(f"Detector 创建失败: {e2}")

check("apriltag（创建 Detector 验证）", _check_apriltag)

section("3. 项目路径 & 协议自测")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

check("仓库根目录可访问",
      lambda: ROOT if os.path.isdir(ROOT) else "找不到仓库根目录")

check("shared/protocol/protocol.py 存在",
      lambda: True if os.path.isfile(
          os.path.join(ROOT, "shared", "protocol", "protocol.py")) else "文件不存在")

check("shared/camera_base.py 存在",
      lambda: True if os.path.isfile(
          os.path.join(ROOT, "shared", "camera_base.py")) else "文件不存在")

check("vision_raspberrypi/src/apriltag_detector.py 存在",
      lambda: True if os.path.isfile(
          os.path.join(ROOT, "vision_raspberrypi", "src", "apriltag_detector.py"))
              else "文件不存在")

def _import_detector():
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "vision_raspberrypi"))
    from vision_raspberrypi.src.apriltag_detector import AprilTagDetector
    from vision_raspberrypi.src.strategy import Strategy, ActionResult
    return "AprilTagDetector & Strategy 可导入"

check("import 生产代码（AprilTagDetector / StrategyEngine）", _import_detector)

section("4. 摄像头检测")

def _list_cameras():
    import cv2
    found = []
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            found.append(f"device_id={idx}  {w}x{h} @ {fps:.0f}fps")
            cap.release()
    if not found:
        return "未检测到任何摄像头（仍可用静态图测试）"
    return "\n" + "\n".join(f"            {f}" for f in found)

warn("可用摄像头扫描（ID 0-4）", _list_cameras)

section("5. 协议自测（快速）")

def _run_protocol_test():
    import subprocess
    result = subprocess.run(
        [sys.executable,
         os.path.join(ROOT, "shared", "protocol", "test_protocol.py")],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        return lines[-1] if lines else "通过"
    else:
        raise RuntimeError(result.stderr[-300:] if result.stderr else "未知错误")

check("shared/protocol/test_protocol.py", _run_protocol_test)

# ──────────────────────────────────────────────────────────
section("结论")
print("""
  如果上面所有 [  OK  ] 均显示，可以直接运行：

    cd apriltag-webcam-demo          # 仓库根目录
    python training/demo_pc/demo.py

  常见 FAIL 修复（建议一次性装全）：
    pip install numpy opencv-python pupil-apriltags pyyaml

  分项修复参考：
    numpy FAIL    → pip install numpy
    cv2 FAIL      → pip install opencv-python
    apriltag FAIL → pip install pupil-apriltags          （Windows 推荐）
                  或 pip install apriltag                 （Linux/macOS）
    yaml FAIL     → pip install pyyaml
    路径 FAIL     → 确认从仓库根目录运行此脚本（cd apriltag-webcam-demo）

  ⚠ Windows 注意：不要用 conda install -c conda-forge apriltag
     该包是 C 库，没有 Python 绑定的 .Detector 接口，
     且 conda remove 时会级联卸载 numpy/mkl 等依赖。
""")
