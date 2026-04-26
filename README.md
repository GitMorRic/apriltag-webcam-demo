# AprilTag Webcam Demo

实时用**电脑摄像头**识别 [Tag36h11 AprilTag](https://github.com/AprilRobotics/apriltag) 二维码，显示 ID、3D 姿态（坐标轴）、距离、偏角，并演示一套简单的机器人决策策略。

适合快速上手 AprilTag 检测算法，无需任何硬件（一台电脑 + 任意摄像头即可）。

---

## 效果预览

```
┌─────────────────────────────────────────────────────────────┐
│ ACTION: PUSH              DETECTED: 2            FPS: 28.5 │
│                                        ┌─────────────────┐ │
│      ┌──────┐                          │ DETECTIONS      │ │
│      │ ID=1 │  ← 蓝色框               │  >> #1  ID=1    │ │
│      │  ↑←  │  ← X/Y/Z 坐标轴         │     D=0.82m     │ │
│      └──────┘                          │     Y=+5.2d     │ │
│                                        └─────────────────┘ │
│                                        ┌─────────────────┐ │
│                                        │  TOP-DOWN VIEW  │ │
│                                        │     ●  (tag)    │ │
│                                        └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ TARGET  ID=1  D=0.82m  Y=+5.2d  CMD  v=+40%  omega=-6%    │
└─────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 环境

```bash
# 推荐 Python 3.10 ~ 3.12，用 conda 管理环境
conda create -n vision python=3.10 -y
conda activate vision

pip install -r training/demo_pc/requirements.txt
```

### 2. 验证环境

```bash
python training/demo_pc/check_env.py
```

### 3. 生成测试二维码图片

```bash
python training/demo_pc/generate_test_tags.py --ids 0 1 2
# 生成到 training/demo_pc/test_images/
```

### 4. 静态图片测试（不需要摄像头）

```bash
python training/demo_pc/test_static.py training/demo_pc/test_images/tag36h11_00001.png
```

### 5. 实时摄像头 demo

```bash
# 在项目根目录运行
python training/demo_pc/demo.py

# 常用参数
python training/demo_pc/demo.py --camera 1        # 第 2 个摄像头
python training/demo_pc/demo.py --tag-size 0.05   # tag 实际边长 5cm
python training/demo_pc/demo.py --alpha 0.4       # 低通滤波强度（越小越平滑）
```

### 6. 按键说明

| 按键 | 功能 |
|------|------|
| `q` / `ESC` | 退出 |
| `s` | 截图保存到 `recordings/` |
| `r` | 开始 / 停止录制视频 |
| `空格` | 暂停 / 继续 |
| `f` | 切换姿态低通滤波 |
| `m` | 切换鸟瞰小地图 |
| `i` | 切换右侧详情面板 |
| `h` | 切换 HUD |

---

## 相机标定（可选，提升距离精度）

```bash
# 打印 9×6 棋盘格（内角点），格边长 25mm
# 运行标定工具，按 空格 采集约 25 张不同角度的图片
python training/demo_pc/calibrate.py --square 0.025
# 标定完成后，把输出的 fx/fy/cx/cy 填入 training/demo_pc/demo.py 的 DEFAULT_CALIB
```

不标定也能运行，只是距离数值误差较大（估算内参）。

---

## 项目结构

```
apriltag-webcam-demo/
├── training/demo_pc/           ← demo 脚本、标定工具、生成测试图
│   ├── demo.py                 ← 主程序：实时识别 + 可视化
│   ├── camera_opencv.py        ← PC 摄像头封装（HAL 层）
│   ├── calibrate_pc.py         ← 相机标定工具
│   ├── generate_test_tags.py   ← 生成测试用 AprilTag 图片
│   ├── check_env.py            ← 环境检查
│   ├── test_static.py          ← 静态图片测试
│   └── requirements.txt
├── shared/
│   ├── visualization.py        ← 可视化模块（HUD / 小地图 / 坐标轴）
│   └── camera_base.py          ← 摄像头 HAL 抽象基类
└── vision_raspberrypi/src/
    ├── apriltag_detector.py    ← AprilTag 检测 + PnP 位姿估计
    └── strategy.py             ← 简单决策层（接近 / 推击 / 避障 / 搜索）
```

### 核心依赖关系

```
demo.py
  ├── camera_opencv.py          继承 shared/camera_base.py
  ├── apriltag_detector.py      调用 pupil-apriltags + OpenCV solvePnP
  │     └── shared/visualization.py   绘制检测结果
  └── strategy.py               根据检测结果决策动作
```

---

## 算法说明

### AprilTag 检测
使用 [pupil-apriltags](https://github.com/pupil-labs/apriltags)（`pupil_apriltags.Detector`）
检测 Tag36h11 族二维码，输出每个 tag 的 ID 和四个角点像素坐标。

### PnP 位姿估计
给定四个角点的图像坐标（2D）和已知物理边长（3D），用
`cv2.solvePnP(SOLVEPNP_IPPE_SQUARE)` 求解相机到 tag 的旋转矩阵 `R` 和平移向量 `t`，
进而得到距离、偏航角（yaw）、俯仰角（pitch）。

### 决策层
`strategy.py` 根据检测到的 tag ID 和距离，按优先级选择目标，
输出动作（PUSH / APPROACH / AVOID / SEARCH / STOP）和速度指令（v%、omega%）。

---

## 许可证

MIT License — 欢迎学习、修改和复用。
