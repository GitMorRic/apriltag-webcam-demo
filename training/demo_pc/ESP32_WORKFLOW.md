# ESP32-S3 开发流程（与 conda/Python 完全不同）

> ESP32 运行的是编译好的 C 二进制固件，没有 Python、没有 conda、没有 apt。
> 开发流程是：**写 C 代码 → 用 ESP-IDF 工具链编译 → 烧录到芯片 → 串口查看日志**。

---

## 和 PC demo / 树莓派的根本区别

| 方面 | PC demo | 树莓派 | ESP32-S3 |
|-----|---------|--------|---------|
| 运行环境 | conda Python | Linux + Python | 裸机，无操作系统（FreeRTOS）|
| 代码语言 | Python | Python | C（ESP-IDF）|
| 包管理 | conda + pip | pip | **无包管理**，源码直接编译进固件 |
| 依赖安装 | `pip install` | `pip install` | 把库源码放 `components/` 目录 |
| 调试方式 | `print()` + 窗口 | `print()` + SSH | 串口打印 `ESP_LOGI()` |
| 代码改了之后 | 直接重新运行 | 直接重新运行 | 重新 `idf.py build flash` |
| AprilTag 库 | `pip install apriltag` | `pip install apriltag` | 把 C 源码放进 `components/apriltag/` |

---

## Step 1：安装 ESP-IDF（一次性）

推荐方式：通过 **VSCode ESP-IDF 插件**安装，避免手动配置环境变量。

### 方式 A：VSCode 插件（推荐）

1. 打开 VSCode → 扩展市场 → 搜索 `ESP-IDF` → 安装 Espressif IDF 插件
2. `Ctrl+Shift+P` → `ESP-IDF: Configure ESP-IDF Extension`
3. 选择 `EXPRESS` → 选版本 `v5.3 LTS` → 点击安装
4. 等待 5-10 分钟（会自动下载工具链、Python 环境等）

安装完成后，VSCode 底部状态栏会出现 ESP-IDF 版本号。

### 方式 B：命令行手动安装

```powershell
# 在 PowerShell 中
# 1. 克隆 ESP-IDF（选一个不含空格的路径，如 C:\esp\esp-idf）
git clone --recursive https://github.com/espressif/esp-idf.git C:\esp\esp-idf --depth=1 -b v5.3

# 2. 运行安装脚本
cd C:\esp\esp-idf
.\install.ps1 esp32s3

# 3. 每次开新终端时激活（或加到 PowerShell profile 里）
. C:\esp\esp-idf\export.ps1
```

验证安装：

```powershell
idf.py --version
# 预期输出：ESP-IDF v5.3.x
```

---

## Step 2：添加 AprilTag 组件（ESP-IDF 的"依赖安装"）

ESP-IDF 没有 pip，依赖库需要以**源码组件**的形式放进 `components/` 目录。

```powershell
cd D:\Engineering\Machinecal\WalkingRobotVision\vision_esp32s3

# 方式 A：git submodule（推荐，方便后续更新）
git submodule add https://github.com/AprilRobotics/apriltag.git components/apriltag

# 然后按 components/README.md 里的模板修改 components/apriltag/CMakeLists.txt
```

> 如果没有 git，也可以直接从 GitHub 下载 zip，解压到 `components/apriltag/`，
> 再手动创建 `CMakeLists.txt`（模板在 `vision_esp32s3/components/README.md`）。

---

## Step 3：修改引脚定义（★每块板子必做）

打开 `vision_esp32s3/main/pin_config.h`，按照实际开发板的接线修改：

```c
// 示例（按实际修改）
#define CAM_PIN_PWDN    -1   // 不使用
#define CAM_PIN_RESET   -1   // 不使用
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD     4   // SCCB SDA
#define CAM_PIN_SIOC     5   // SCCB SCL
#define CAM_PIN_D7      18
// ... 其余引脚
#define UART1_TX_PIN    43
#define UART1_RX_PIN    44
```

---

## Step 4：首次编译

```powershell
# 激活 ESP-IDF 环境（如果是命令行方式安装）
. C:\esp\esp-idf\export.ps1

# 进入工程目录
cd D:\Engineering\Machinecal\WalkingRobotVision\vision_esp32s3

# 设置目标芯片
idf.py set-target esp32s3

# 可选：检查关键配置（PSRAM Octal + CPU 240MHz）
idf.py menuconfig
# 路径：Component config → ESP32S3-Specific → CPU frequency: 240MHz
#       Component config → ESP PSRAM → SPI RAM config → Mode: Octal

# 编译（首次约 3-5 分钟）
idf.py build
```

**编译成功的标志**：

```
...
[100%] Linking CXX executable vision_esp32s3.elf
...
Project build complete. To flash, run:
  idf.py flash
or
  python -m esptool ...
```

如果编译失败，先看错误的最后 20 行，常见原因：

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `No such file or directory: apriltag.h` | apriltag 组件未添加 | 完成 Step 2 |
| `undefined reference to 'apriltag_detector_create'` | CMakeLists.txt 未正确配置 | 检查 components/apriltag/CMakeLists.txt |
| `PSRAM initialization failed` | sdkconfig 中 PSRAM 模式不对 | menuconfig 中改为 Octal |

---

## Step 5：烧录到开发板

```powershell
# 用 USB 连接 ESP32-S3 开发板（USB-OTG 或 UART-USB 口）

# 查看串口号（Windows）
Get-WMIObject Win32_SerialPort | Select-Object Name, DeviceID
# 预期：USB Serial Device (COM3) 类似的输出

# 烧录并打开串口监视器
idf.py -p COM3 flash monitor
# COM3 改成你看到的实际串口号

# 如果识别不到串口：
# 1. 按住 BOOT 键 → 按 RESET 键 → 松 BOOT → 重新运行 flash
# 2. 安装 CP2102 或 CH340 驱动（取决于开发板的 USB 转串口芯片）
```

**正常启动日志**：

```
I (312) main: CPU freq: 240 MHz
I (318) camera_task: OV5640 init OK, 640x480 GRAYSCALE
I (420) detector_task: apriltag detector ready, family=tag36h11
I (421) main: 主循环启动，目标 20Hz
I (471) main: loop#1 detect=0 pose_ok=0 action=5(SEARCH)
I (521) main: loop#2 detect=0 pose_ok=0 action=5(SEARCH)
...
```

把 AprilTag 放到摄像头前后：

```
I (4312) main: loop#76 detect=1 pose_ok=1 action=2(PUSH)
I (4312) uart_task: TX Vision frame seq=75 tag_id=1 yaw=+5 dist=85
```

---

## Step 6：修改代码 → 重新编译 → 烧录

这是 ESP32 开发的日常循环，与 Python 不同，每次修改都需要重新编译：

```powershell
# 修改代码后
idf.py build && idf.py -p COM3 flash monitor

# 只烧录不重新编译（固件未变）
idf.py -p COM3 flash

# 只看串口不烧录
idf.py -p COM3 monitor
# 退出 monitor：按 Ctrl+]
```

---

## Step 7：串口日志解读

ESP-IDF 的日志格式：

```
I (时间ms) 模块名: 日志内容
W (时间ms) 模块名: 警告
E (时间ms) 模块名: 错误
```

| 日志内容 | 含义 | 是否正常 |
|---------|------|---------|
| `OV5640 init OK` | 摄像头初始化成功 | ✅ |
| `Camera capture failed` | 帧读取失败 | ❌ 检查接线 |
| `detect=0` | 当前帧没有识别到 tag | ⚠️（正常，没有 tag 时）|
| `detect=1 pose_ok=1` | 识别到 1 个 tag，位姿解算成功 | ✅ |
| `PSRAM initialization failed` | PSRAM 初始化失败（N16R8 上不应出现）| ❌ 检查 sdkconfig |
| `Stack overflow in task` | FreeRTOS 任务栈溢出 | ❌ 增大对应任务的栈大小 |

---

## ESP32 vs 树莓派 vs PC：算法一致性验证

三个平台的检测和决策逻辑是对应的：

```
PC demo（Python）                树莓派（Python）            ESP32（C）
─────────────────────────────────────────────────────────────────────
apriltag_detector.py             apriltag_detector.py       detector_task.c
  ↓ 同一套 PnP 公式                ↓                           ↓ pose_task.c
strategy.py                      strategy.py                strategy.c
  ↓ 同一套决策逻辑                  ↓                           ↓
protocol.py（pack）               protocol.py（pack）         protocol.c（pack）
  ↓ 同一个帧格式                    ↓                           ↓
（打印结果，不发串口）              串口发送                    串口发送
```

验证三者一致的方法：
1. 用同一张 AprilTag 图片，分别在 PC demo 和树莓派上运行
2. 对比打印出的 yaw_deg 和 dist 值
3. 误差应 < 0.5°（来自浮点精度），如有较大差异，说明内参或 tag_size 配置不一致

---

## 总结：两套开发环境并行

```
conda activate vision          idf.py build
      │                              │
      ▼                              ▼
Python 生态（PC/RPi）           ESP-IDF（ESP32）
  pip install xxx               组件放 components/
  修改后直接运行                 修改后重新编译烧录
  print() 调试                  ESP_LOGI() + monitor
  cv2.imshow() 可视化           只有串口文本
```

**建议新人策略**：先在 PC demo 上跑通算法逻辑，确认 tag 识别和决策都对，再上 ESP32 调试。因为 PC 上迭代速度快 10 倍以上。
