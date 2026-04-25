# PC 端 AprilTag 演示

> 在你自己的笔记本上，用电脑摄像头实时识别 AprilTag，**无需任何机器人硬件**。
>
> 代码直接复用 `vision_raspberrypi/src/` 中的检测和决策模块，零改动，仅替换摄像头驱动。

---

## 为什么做这个 demo

1. **零硬件门槛**：不需要树莓派、ESP32、OV5640，只需一台装了 Python 的电脑
2. **直接看生产代码跑起来**：`AprilTagDetector` 和 `StrategyEngine` 与比赛时的代码完全一致
3. **理解 HAL 设计**：通过对比 `camera_opencv.py` 和 `camera_manager.py`，直观理解抽象接口的价值

---

## 快速上手（5 分钟）

### 1. 安装依赖

```bash
# 在仓库根目录执行
pip install -r training/demo_pc/requirements.txt
```

> Windows 用户如果 `apriltag` 安装失败，尝试：
> ```
> pip install apriltag --find-links https://github.com/duckietown/apriltag/releases
> ```

### 2. 打印或显示一个 AprilTag

从这里下载 Tag36h11 的图片，打印出来或在手机/另一屏幕上显示：

- 标准 tag 图片集：https://github.com/AprilRobotics/apriltag-imgs/tree/master/tag36h11
- 建议使用 `tag36h11_00000.png`（ID=0，中立）、`tag36h11_00001.png`（ID=1，蓝方）、`tag36h11_00002.png`（ID=2，黄方）

### 3. 运行 demo

```bash
# 在仓库根目录执行（必须在根目录，否则路径解析失败）
cd D:\Engineering\Machinecal\WalkingRobotVision

python training/demo_pc/demo.py
```

看到摄像头画面后，把打印好的 tag 举在摄像头前。

---

## 操作按键

| 按键 | 功能 |
|-----|------|
| `q` | 退出 |
| `s` | 保存当前帧截图（`saved_frame_XXXXXX.jpg`）|
| `空格` | 暂停 / 继续 |

---

## 可选参数

```bash
# 使用第 2 个摄像头（USB 外接）
python training/demo_pc/demo.py --camera 1

# 修改 tag 物理边长（单位：米，默认 0.12）
python training/demo_pc/demo.py --tag-size 0.10

# 设置我方队伍 ID（1=蓝，2=黄）
python training/demo_pc/demo.py --our-team 2
```

---

## 画面说明

```
┌──────────────────────────────────────────────────────┐
│ ACTION: PUSH                                FPS:28.5 │  ← 决策状态
│                                                      │
│      ┌──────────┐                                    │
│      │  ID:1    │ ← 检测框（绿色）                    │
│      │  ●       │ ← 中心点（红色）                    │
│      │  ↑ ← →   │ ← 坐标轴（RGB = XYZ）              │
│      └──────────┘                                    │
│                                                      │
│ tag_id=1  yaw=+12.3deg  dist=0.45m  reason: push    │  ← 底部信息
└──────────────────────────────────────────────────────┘
```

| 颜色 | 含义 |
|-----|------|
| 绿色框 | 检测到的 tag 边界 |
| 红色圆点 | tag 中心点 |
| 红箭头 | X 轴（tag 右方）|
| 绿箭头 | Y 轴（tag 下方）|
| 蓝箭头 | Z 轴（tag 朝向摄像头，指向屏幕外）|
| 顶部文字 | 当前决策 action 和帧率 |
| 底部文字 | tag 详细信息 |

---

## 距离测量精度说明

demo 使用笔记本摄像头的**估算内参**（fx=fy=640），实际误差约 10-20%。

如果需要精确的距离测量（用于调试策略阈值），可以对你的笔记本摄像头做标定：

```bash
# 用同一套棋盘格标定工具
python vision_raspberrypi/tools/calibrate.py --live \
  --output my_laptop_640x480.yaml

# 然后把标定结果更新到 demo.py 中的 DEFAULT_CALIB 字典
```

---

## 代码架构说明（核心学习点）

```
demo.py
  ├── OpenCVCamera(CameraBase)      ← 只有这一个文件是 PC 特有的
  │     camera_opencv.py
  │
  ├── AprilTagDetector              ← 直接 import 生产代码，零改动
  │     vision_raspberrypi/src/apriltag_detector.py
  │
  └── StrategyEngine                ← 直接 import 生产代码，零改动
        vision_raspberrypi/src/strategy.py
```

对比 `camera_opencv.py` 和 `vision_raspberrypi/src/camera_manager.py`，可以看到：

- 两者都继承自 `shared/camera_base.py` 中的 `CameraBase`
- 两者都实现了相同的 `open()` / `read()` / `close()` 接口
- `read()` 都返回 `(bool, bgr_frame)`
- **上层代码（detector / strategy）对两者无感知**

这就是"硬件抽象层（HAL）"的实际意义。

---

## 常见问题

| 问题 | 解决方案 |
|-----|---------|
| 摄像头无法打开 | 检查其他程序（钉钉/腾讯会议）是否占用摄像头，关闭后重试 |
| `apriltag` 安装失败 | Windows 上尝试 `conda install -c conda-forge apriltag` |
| 画面黑屏但不报错 | 尝试 `--camera 1` 换一个摄像头 ID |
| 能识别 tag 但距离明显不对 | 需要对你的摄像头做标定，见上文"距离测量精度说明" |
| 识别不稳定/频繁丢失 | 改善光照，或把 `quad_decimate` 从 1.0 调低（但 PC 上通常不需要）|
