"""
training/demo_pc/generate_test_tags.py
本地生成 Tag36h11 测试图片 —— 完全不需要网络。

工作原理：
  使用 OpenCV 内置的 cv2.aruco.DICT_APRILTAG_36H11 字典直接渲染
  AprilTag 图片，与 pupil_apriltags 检测器使用同一份编码规范，
  100% 保证生成的图片可以被检测到。

  （opencv-python >= 4.7 已内置 aruco 模块，无需额外安装）

运行方式：
  cd WalkingRobotVision
  python training/demo_pc/generate_test_tags.py           # 生成 ID 0/1/2
  python training/demo_pc/generate_test_tags.py --ids 0 1 2 5 10
  python training/demo_pc/generate_test_tags.py --show    # 生成后弹窗预览
  python training/demo_pc/generate_test_tags.py --verify  # 回测检测

生成的文件保存到：
  training/demo_pc/test_images/tag36h11_XXXXX.png
"""

import sys
import os
import argparse
import logging

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vision_raspberrypi"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("tag_gen")


# ──────────────── 核心生成逻辑 ────────────────

def generate_tag(tag_id: int, marker_size: int = 400, border_bits: int = 1) -> np.ndarray:
    """
    使用 cv2.aruco 生成单张 Tag36h11 灰度图片。

    cv2.aruco.DICT_APRILTAG_36H11 是 OpenCV 内置字典，
    与 pupil_apriltags / apriltag C 库使用完全相同的编码规范。

    Args:
        tag_id:      0=中立 / 1=蓝方 / 2=黄方 及以上（最大 586）
        marker_size: 生成的标记图片尺寸（像素，不含边距，默认 400）
        border_bits: 白色安静区宽度（以 cell 为单位，默认 1）

    Returns:
        含白色边距的灰度图（np.ndarray uint8）
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)

    # 生成裸标记（无边距）
    tag_img = cv2.aruco.generateImageMarker(dictionary, tag_id, marker_size)

    # 添加白色安静区（"quiet zone"）
    # apriltag 官方建议至少留 2 个 cell 的白边，这里用像素计
    # cell_size = marker_size / 10（tag36h11 是 10×10 cell）
    cell_size  = marker_size // 10
    pad        = cell_size * 3     # 3 个 cell 的白色边距（充足）
    h, w       = tag_img.shape
    padded     = np.full((h + 2 * pad, w + 2 * pad), 255, dtype=np.uint8)
    padded[pad:pad + h, pad:pad + w] = tag_img

    return padded


def save_tag_image(tag_id: int, save_dir: str,
                   marker_size: int = 400) -> str:
    """生成并保存一张 tag 图片，返回保存路径。"""
    os.makedirs(save_dir, exist_ok=True)
    img  = generate_tag(tag_id, marker_size)
    path = os.path.join(save_dir, f"tag36h11_{tag_id:05d}.png")
    cv2.imwrite(path, img)
    log.info("已保存：%s  (%dx%d px)", path, img.shape[1], img.shape[0])
    return path


# ──────────────── 回测验证 ────────────────

def verify_tag(img_path: str, expected_id: int) -> bool:
    """用 AprilTagDetector 反向验证生成的图片能被正确检测。"""
    from vision_raspberrypi.src.apriltag_detector import AprilTagDetector

    calib  = {"fx": 640.0, "fy": 640.0, "cx": 320.0, "cy": 320.0,
               "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]}
    detect = {"family": "tag36h11", "tag_size": 0.12,
               "nthreads": 1, "quad_decimate": 1.0,
               "quad_sigma": 0.0, "refine_edges": True}

    gray    = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        log.error("图片读取失败：%s", img_path)
        return False

    bgr     = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    det     = AprilTagDetector(calib, detect)
    results = det.detect(bgr)

    if results:
        r  = results[0]
        ok = r.tag_id == expected_id
        sym = "✓" if ok else f"✗ 检测到 ID={r.tag_id}，期望 ID={expected_id}"
        log.info("  验证 %s  dist=%.2fm  yaw=%+.1f°  conf=%.0f",
                 sym, r.distance, r.yaw_deg, r.confidence)
        return ok
    else:
        log.warning("  验证 ✗  未检测到任何 tag")
        return False


# ──────────────── 入口 ────────────────

def main():
    parser = argparse.ArgumentParser(
        description="本地生成 Tag36h11 测试图片（完全离线，cv2.aruco）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python training/demo_pc/generate_test_tags.py
  python training/demo_pc/generate_test_tags.py --ids 0 1 2 5 10 50
  python training/demo_pc/generate_test_tags.py --marker-size 600 --show
  python training/demo_pc/generate_test_tags.py --verify
        """,
    )
    parser.add_argument("--ids",         type=int, nargs="+", default=[0, 1, 2],
                        help="要生成的 tag ID 列表（默认 0 1 2）")
    parser.add_argument("--marker-size", type=int, default=400,
                        help="标记图片尺寸（像素，不含白色边距，默认 400）")
    parser.add_argument("--output-dir",  type=str, default=None,
                        help="输出目录（默认 training/demo_pc/test_images/）")
    parser.add_argument("--show",    action="store_true",
                        help="生成后弹出预览窗口（任意键=下一张，q=退出）")
    parser.add_argument("--verify",  action="store_true",
                        help="生成后用 AprilTagDetector 回测验证是否可检测")
    args = parser.parse_args()

    save_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "test_images"
    )

    print(f"\n  输出目录：{save_dir}")
    print(f"  生成 IDs：{args.ids}")
    print(f"  标记尺寸：{args.marker_size} px\n")

    saved, failed = [], []
    for tag_id in args.ids:
        try:
            log.info("生成 tag36h11 ID=%d ...", tag_id)
            path = save_tag_image(tag_id, save_dir, args.marker_size)
            saved.append((tag_id, path))
        except Exception as e:
            log.error("ID=%d 生成失败：%s", tag_id, e)
            failed.append(tag_id)

    print(f"\n  生成完成：{len(saved)}/{len(args.ids)} 成功\n")
    if failed:
        print(f"  失败 ID：{failed}\n")

    # 可选：回测检测
    if args.verify and saved:
        print("  验证（AprilTagDetector 回测）...\n")
        pass_n = sum(verify_tag(p, tid) for tid, p in saved)
        print(f"\n  验证结果：{pass_n}/{len(saved)} 通过\n")

    # 可选：弹窗预览
    if args.show and saved:
        for tag_id, path in saved:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            title = f"tag36h11 ID={tag_id}  (任意键=下一张  q=退出)"
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 600, 600)
            cv2.imshow(title, img)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                break

    if saved:
        print("  接下来可以运行静态图检测测试：")
        print(f"    python training/demo_pc/test_static.py --dir {save_dir}\n")


if __name__ == "__main__":
    main()
