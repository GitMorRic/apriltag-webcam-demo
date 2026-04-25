"""
training/demo_pc/test_static.py
用静态图片测试 AprilTag 识别算法 —— 无需摄像头。

用法：
  # 测试单张图片
  python training/demo_pc/test_static.py --image path/to/tag.jpg

  # 批量测试一个目录下的所有图片
  python training/demo_pc/test_static.py --dir path/to/images/

  # 自动从网络下载官方测试图（优先 jsDelivr，国内可用；失败则本地生成）
  python training/demo_pc/test_static.py --download

  # 不指定参数时，脚本会尝试在当前目录/test_images/ 搜索 .jpg/.png
  python training/demo_pc/test_static.py
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

from vision_raspberrypi.src.apriltag_detector import AprilTagDetector, DetectionResult
from vision_raspberrypi.src.strategy import Strategy

logging.basicConfig(level=logging.WARNING)   # 静默库日志，只看结果

# ── 与 demo.py 保持一致的默认内参 ──
DEFAULT_CALIB = {
    "fx": 640.0, "fy": 640.0,
    "cx": 320.0, "cy": 240.0,
    "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
}
DEFAULT_DETECT = {
    "family": "tag36h11", "tag_size": 0.12,
    "nthreads": 2, "quad_decimate": 1.0,
    "quad_sigma": 0.0, "refine_edges": True,
}
DEFAULT_STRATEGY = {
    "our_team_id": 1, "enemy_team_id": 2,
    "push_min_dist_m": 0.3, "push_max_dist_m": 3.0,
    "push_angle_threshold_deg": 15.0,
    "protect_radius_m": 0.5, "lost_timeout_s": 1.0,
}

ACTION_NAMES = {0:"STOP", 1:"VELOCITY", 2:"PUSH", 3:"AVOID", 4:"PROTECT", 5:"SEARCH"}


def process_image(path: str, detector: AprilTagDetector,
                  engine: Strategy, show: bool = True) -> bool:
    """
    处理单张图片，打印结果并可选显示带标注的图像。
    Returns: 是否检测到至少一个 tag
    """
    img = cv2.imread(path)
    if img is None:
        print(f"  ✗ 无法读取图片：{path}")
        return False

    h, w = img.shape[:2]

    # ── 检测 ──
    detections = detector.detect(img)

    # ── 决策 ──
    action = engine.decide(detections)

    # ── 打印结果 ──
    fname = os.path.basename(path)
    print(f"\n{'─'*50}")
    print(f"  图片  : {fname}  ({w}x{h})")
    print(f"  检测数: {len(detections)} 个 tag")

    if detections:
        for i, r in enumerate(detections):
            print(f"\n  [Tag {i+1}]")
            print(f"    tag_id     = {r.tag_id}  "
                  f"({'中立' if r.tag_id==0 else '蓝方' if r.tag_id==1 else '黄方' if r.tag_id==2 else '未知'})")
            print(f"    距离       = {r.distance:.3f} m")
            print(f"    水平偏角   = {r.yaw_deg:+.2f}°  (正=右偏，负=左偏)")
            print(f"    俯仰角     = {r.pitch_deg:+.2f}°")
            print(f"    图像中心   = ({r.center[0]:.1f}, {r.center[1]:.1f}) px")
            print(f"    置信度     = {r.confidence:.1f}")
            tv = r.pose_t
            print(f"    tvec (m)   = [{tv[0]:+.4f},  {tv[1]:+.4f},  {tv[2]:+.4f}]")
        action_name = ACTION_NAMES.get(action.action, str(action.action))
        print(f"\n  → 决策: {action_name}  v={action.v_pct:.0f}%  ω={action.omega_pct:.0f}%")
        print(f"    原因: {action.reason}")
    else:
        print("  → 未检测到 tag")

    # ── 可视化 & 显示 ──
    annotated = detector.draw_detections(img, detections)

    if show:
        win_title = f"{fname}  [任意键=下一张  q=退出]"
        cv2.imshow(win_title, annotated)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(win_title)
        if key == ord('q'):
            return None   # 信号：用户要退出

    # 保存标注图
    out_path = os.path.splitext(path)[0] + "_annotated.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"  已保存标注图：{out_path}")

    return len(detections) > 0


def download_test_images(save_dir: str):
    """
    尝试从多个 CDN 下载 Tag36h11 ID 0/1/2 测试图。
    如果全部失败（网络问题），自动 fallback 到本地生成。

    CDN 优先级：
      1. jsDelivr（国内可访问，推荐）
      2. GitHub raw（境外可访问）
    """
    import urllib.request

    # jsDelivr 国内可直接访问（国内首选），GitHub raw 境外用
    CDN_LIST = [
        "https://cdn.jsdelivr.net/gh/AprilRobotics/apriltag-imgs@master/tag36h11/",
        "https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/master/tag36h11/",
    ]
    files = ["tag36h11_00000.png", "tag36h11_00001.png", "tag36h11_00002.png"]

    os.makedirs(save_dir, exist_ok=True)
    print(f"正在下载测试图到 {save_dir} ...")
    print("  （先尝试 jsDelivr CDN，国内网络可访问）\n")

    downloaded = []
    for fname in files:
        dest = os.path.join(save_dir, fname)
        success = False
        for base_url in CDN_LIST:
            url = base_url + fname
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"  ✓ {fname}  (来源: {base_url.split('/')[2]})")
                downloaded.append(dest)
                success = True
                break
            except Exception as e:
                last_err = e
        if not success:
            print(f"  ✗ {fname}  →  所有 CDN 均失败: {last_err}")

    if len(downloaded) < len(files):
        missing = len(files) - len(downloaded)
        print(f"\n  {missing} 张下载失败（通常是网络问题）。")
        print("  ─────────────────────────────────────────────────────────")
        print("  方案 A（推荐，完全离线）：本地生成，直接运行：")
        print("    python training/demo_pc/generate_test_tags.py")
        print()
        print("  方案 B：手动从浏览器下载后放到以下目录：")
        print(f"    {save_dir}")
        print("    文件名：tag36h11_00000.png  tag36h11_00001.png  tag36h11_00002.png")
        print("    jsDelivr 直链（浏览器可访问）：")
        for fname in files:
            if not os.path.isfile(os.path.join(save_dir, fname)):
                print(f"    https://cdn.jsdelivr.net/gh/AprilRobotics/apriltag-imgs"
                      f"@master/tag36h11/{fname}")
        print("  ─────────────────────────────────────────────────────────")

        # 如果一张都没下到，尝试自动 fallback 到本地生成
        if not downloaded:
            print("\n  未下载到任何图片，尝试本地自动生成...")
            try:
                gen_dir = os.path.dirname(__file__)
                gen_script = os.path.join(gen_dir, "generate_test_tags.py")
                if os.path.isfile(gen_script):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "generate_test_tags", gen_script)
                    gen_mod = importlib.util.load_from_spec(spec)  # type: ignore
                    spec.loader.exec_module(gen_mod)  # type: ignore
                    for tid in [0, 1, 2]:
                        path = gen_mod.save_tag_image(tid, save_dir, pixel_size=60)
                        downloaded.append(path)
                    print("  本地生成成功！\n")
            except Exception as e:
                print(f"  本地生成也失败: {e}")
                print("  请手动运行：python training/demo_pc/generate_test_tags.py\n")

    return downloaded


def collect_images(paths: list) -> list:
    """从路径列表（文件/目录）中收集所有图片路径。"""
    result = []
    for p in paths:
        if os.path.isfile(p):
            result.append(p)
        elif os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    result.append(os.path.join(p, fn))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="AprilTag 静态图片测试（无需摄像头）"
    )
    parser.add_argument("--image",    type=str,  help="单张图片路径")
    parser.add_argument("--dir",      type=str,  help="图片目录路径")
    parser.add_argument("--download", action="store_true",
                        help="下载官方测试图（jsDelivr CDN，国内可用）；失败则自动本地生成")
    parser.add_argument("--no-show",  action="store_true",
                        help="不弹出图片窗口（只打印结果）")
    parser.add_argument("--tag-size", type=float, default=0.12,
                        help="tag 物理边长（米，默认 0.12）")
    args = parser.parse_args()

    # ── 初始化 ──
    detect_cfg = dict(DEFAULT_DETECT)
    detect_cfg["tag_size"] = args.tag_size
    detector = AprilTagDetector(DEFAULT_CALIB, detect_cfg)
    engine   = Strategy(DEFAULT_STRATEGY)

    # ── 收集图片路径 ──
    image_paths = []

    if args.download:
        dl_dir = os.path.join(os.path.dirname(__file__), "test_images")
        image_paths = download_test_images(dl_dir)
    elif args.image:
        image_paths = [args.image]
    elif args.dir:
        image_paths = collect_images([args.dir])
    else:
        # 自动在默认位置搜索
        default_dirs = [
            os.path.join(os.path.dirname(__file__), "test_images"),
            os.getcwd(),
        ]
        image_paths = collect_images(default_dirs)
        if not image_paths:
            print("没有找到测试图片。")
            print("用法示例：")
            print("  python training/demo_pc/test_static.py --download")
            print("  python training/demo_pc/test_static.py --image my_tag.jpg")
            return

    if not image_paths:
        print("没有可测试的图片文件。")
        return

    print(f"\n共找到 {len(image_paths)} 张图片，开始测试...\n")

    total = len(image_paths)
    hits  = 0

    for path in image_paths:
        result = process_image(path, detector, engine, show=not args.no_show)
        if result is None:    # 用户按 q 退出
            break
        if result:
            hits += 1

    cv2.destroyAllWindows()

    print(f"\n{'═'*50}")
    print(f"  测试结果：{hits}/{total} 张图片检测到 tag")
    if hits == total:
        print("  ✓ 全部通过！算法运行正常。")
    elif hits > 0:
        print("  ⚠ 部分通过。未检测到 tag 的图片可能：")
        print("    1. 图片太小/模糊/光线不足")
        print("    2. quad_decimate 需要调整（对小 tag 降为 1.0）")
        print("    3. 图片中没有 Tag36h11 族系的 tag")
    else:
        print("  ✗ 全部未检测到 tag。请检查：")
        print("    1. 图片里是否真的有 AprilTag（Tag36h11 族）")
        print("    2. apriltag 库是否安装正确（运行 check_env.py 验证）")


if __name__ == "__main__":
    main()
