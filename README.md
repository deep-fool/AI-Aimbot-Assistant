# 🎯 AI Aim Assist — 基于 TensorRT 的实时目标检测与辅助瞄准系统

<div align="center">

[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.16-orange.svg)](https://developer.nvidia.com/tensorrt)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.24-lightgrey.svg)](https://onnxruntime.ai/)
[![Platform](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue.svg)](https://www.microsoft.com/windows)
[![OpenCV](https://img.shields.io/badge/OpenCV-latest-red.svg)](https://opencv.org/)

</div>

---

## 📑 目录

- [项目简介](#项目简介)
- [核心优势](#核心优势)
- [功能特性](#功能特性)
- [架构设计](#架构设计)
- [依赖库一览](#依赖库一览)
- [编译指南](#编译指南)
- [模型准备](#模型准备)
- [使用说明](#使用说明)
- [配置参数](#配置参数)
- [项目结构](#项目结构)

---

## 项目简介

一套**超低延迟**的 AI 视觉辅助瞄准系统，基于 **NVIDIA TensorRT** / **ONNX Runtime** 进行 GPU 加速的 YOLO 目标检测推理，配合 **DXGI Desktop Duplication API** 实现微秒级屏幕截图，辅以**自适应 PID 控制器**与**贝塞尔曲线运动模拟**，完成从"看见"到"瞄准"的完整闭环。

> 🚀 从截图 → 推理 → 后处理 → 移动到位的全链路延迟控制在 **3ms** 级别。

---

## 核心优势

### ⚡ 截图速度极快

| 截图后端 | 技术 | 延迟 | 说明 |
|---------|------|------|------|
| **DXGI** | `IDXGIOutputDuplication` | **< 1ms** | 直接从 GPU 显存复制桌面帧，零 CPU 拷贝，首选方案 |
| **WGC** | `Windows.Graphics.Capture` | < 2ms | WinRT 现代截图 API，兼容性好 |
| **GDI** | `BitBlt` | < 5ms | CPU 管线，仅作降级兜底 |

**性能秘诀**：
- **SIMD 加速** BGRA→BGR 颜色空间转换，充分利用 AVX2 指令集
- **预分配内存池** (`PreallocatedMemoryPool`)：从 128×128 到 1920×1080 的缓冲区预分配，零动态分配
- **BMP 头缓存**：捕获循环内零格式转换开销
- **纹理对缓存** (`TexturePair` map)：按宽高缓存 D3D11 纹理对，避免重复创建释放
- 截图线程与推理线程**并行流水线**工作，截图速度不受推理耗时影响

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Capture    │ ──▶ │  Inference  │ ──▶ │  Aim + Move │
│  Thread     │     │  Thread     │     │  Main Loop  │
│  ~4000 FPS  │     │  TensorRT   │     │  PID+Bezier │
└─────────────┘     └─────────────┘     └─────────────┘
```

### 🧠 TensorRT 推理加速

- 支持 **YOLOv10**（输出 `[1, 300, 6]` 绝对坐标）和 **YOLOv11**（输出 `[1, 4+nc, anchors]` 相对坐标）两种模型格式
- **自动检测**模型类型：根据 Engine 输出张量形状自动适配后处理逻辑，无需手动配置
- **FP16 精度**推理：在几乎不损失检测精度的情况下，推理速度翻倍
- **CUDA Stream 异步流水线**：H2D 拷贝、推理、D2H 拷贝全部异步并行
- 典型推理耗时：**< 2ms**（224×224 输入，RTX 2060+ 显卡）

```
传统 CPU 推理（ONNX）:  ████████████████  15–30ms
TensorRT + CUDA (本方案): ██                < 2ms
                                ↑ 快 10–15 倍
```

### 🎯 智能瞄准控制

- **双重 PID 控制**：X/Y 轴独立 PID 控制器，带 Taylor 级数预测（速度+加速度）
- **距离分区速度限制**：近/中/远/超远四级分区，近距离微调（≤3px/tick），远距离快速逼近（≤110px/tick）
- **振荡抑制**：自动检测方向翻转，施加衰减系数防止过冲抖动
- **贝塞尔曲线移动**：长距离移动时模拟人手自然曲线轨迹，带随机偏移
- **EMA 目标速度预测**：指数移动平均跟踪目标运动速度，预判射击提前量
- **子像素累积**：分数移动量累加到整数阈值才执行，确保平滑追踪

---

## 功能特性

### 🔍 AI 识别
- [x] YOLOv10 / YOLOv11 实时目标检测
- [x] TensorRT Engine 与 ONNX Runtime 双推理后端，一键切换
- [x] 动态模型热重载（运行时更换模型无需重启）
- [x] 置信度阈值可调 + NMS 去重
- [x] 类别过滤（指定检测类别 ID）
- [x] 自适应 Letterbox 缩放预处理
- [x] 目标丢失自动重捕获（最近优先）

### 🖥️ 屏幕捕获
- [x] DXGI / WGC / GDI 三种捕获后端，自动降级
- [x] 可调捕获半径（112px 默认 → 224×224 区域）
- [x] 捕获区域运行时动态调整（无锁安全切换）
- [x] DLL 导出接口，可嵌入任意宿主程序

### 🎮 瞄准辅助
- [x] 自适应 PID 控制（Kp/Ki/Kd 实时可调）
- [x] Taylor 级数运动预测（位置 + 速度 + 加速度）
- [x] 自适应 EMA 滤波（低通/速度/加速度三级）
- [x] 距离分区步长限制
- [x] 振荡检测与自动抑制
- [x] 贝塞尔曲线类人移动
- [x] 初始斜坡缓启动（0.75× → 1.0×，150ms）
- [x] 瞄准偏移量（百分比调整弹着点）
- [x] 死区设置

### 🔫 自动扳机
- [x] 目标进入开火半径自动触发（默认 3px）
- [x] 可独立开关

### 🕹️ 反横拉辅助 (Counter-Strafe)
- [x] 检测 A/D/W/S 按键，自动注入反向按键
- [x] KMBox 硬件注入 / 系统 SendInput 两种后端
- [x] Raw Input 检测避免反馈循环
- [x] 松开扳机或丢失目标时自动释放

### 📟 KMBox 硬件支持
- [x] TCP/IP 网络通信协议（加密传输）
- [x] 硬件级鼠标/键盘注入（绕过反作弊检测）
- [x] 物理鼠标状态监测与屏蔽
- [x] 断连自动降级为系统 API

### 🎛️ ImGui 控制面板
- [x] 7 标签页：Monitor / PID / Aim / Advanced / FindColor / Device / Config
- [x] 实时 FPS / 置信度 / 距离 / 速度 / 锁定状态监控
- [x] 折线图实时绘制（速度、误差变化）
- [x] 全部参数滑块/输入框实时调节
- [x] 配置保存/加载（INI 格式）
- [x] DirectX 11 渲染，游戏内叠加显示

### 🌈 HSV 找色瞄准（辅助模式）
- [x] 双 HSV 范围配置（适应不同敌人颜色）
- [x] ROI 限定扫描区域
- [x] 轮廓检测 + 最小面积过滤
- [x] 可配合 AI 检测或独立使用

---

## 架构设计

```
                          ┌──────────────────────┐
                          │     ImGui UI Thread   │
                          │   (DX11 Overlay 渲染)  │
                          └──────────┬───────────┘
                                     │ 参数读写
                          ┌──────────▼───────────┐
                          │     Main Thread       │
                          │  - PID/Bezier 计算    │
                          │  - 鼠标移动执行       │
                          │  - Counter-Strafe     │
                          │  - Auto Fire          │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
   ┌──────────▼──────────┐ ┌────────▼────────┐ ┌──────────▼──────────┐
   │  Capture Thread     │ │  Display Thread  │ │   KMBox / SendInput │
   │                     │ │                  │ │                     │
   │  DXGI / GDI / WGC   │ │  TensorRT / ONNX │ │  鼠标/键盘硬件注入   │
   │  → BMP 帧缓冲       │ │  → Detection[]   │ │  网络加密通信       │
   │  ~4000 FPS          │ │  → NMS 后处理    │ │                     │
   └─────────────────────┘ └─────────────────┘ └─────────────────────┘
```

### 线程安全模型

采用**确定性握手协议**替代传统的 sleep 等待：

1. 主控线程设置 `g_running = false` 发出停车信号
2. 工作线程在安全点停车后设置各自的 `parked` 标志
3. 主控线程 `WaitAllParked()` 阻塞等待全部线程停止
4. 安全修改尺寸/资源/模型后，`g_running = true` 放行

> 这确保了改截图区域、重载模型等操作 100% 不会导致卡死或竞争条件。

---

## 依赖库一览

### 推理引擎

| 库 | 版本 | 用途 | 下载 |
|---|------|------|------|
| **NVIDIA TensorRT** | 10.16.0.72 | GPU 推理加速 (.engine) | [NVIDIA Developer](https://developer.nvidia.com/tensorrt) |
| **ONNX Runtime GPU** | 1.24.4 | 跨平台推理备选 (.onnx) | [GitHub Releases](https://github.com/microsoft/onnxruntime/releases) |
| **CUDA Toolkit** | 12.8 | GPU 计算平台 | [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) |

### 计算机视觉

| 库 | 版本 | 用途 | 下载 |
|---|------|------|------|
| **OpenCV** | 4.x+ | 图像预处理/后处理/显示 | [OpenCV Releases](https://opencv.org/releases/) |

### 图形与截图

| 库 | 用途 |
|---|------|
| **DirectX 11** (d3d11.lib) | DXGI 截图 + ImGui 渲染 |
| **DXGI 1.2** (dxgi.lib) | GPU 桌面复制 API |
| **Windows GDI** | 传统截图兜底方案 |

### GUI

| 库 | 用途 |
|---|------|
| **Dear ImGui** | 游戏内控制面板（源码编译进项目） |

### 网络与系统

| 库 | 用途 |
|---|------|
| **Winsock2** | KMBox TCP 通信 |
| **WinMM** (winmm.lib) | 高精度定时器 (`timeBeginPeriod(1)`) |

### 集成方式

```
包含目录 (Include):
├── E:\TensorRT-10.16.0.72\include          ← TensorRT SDK
├── C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include  ← CUDA SDK
├── D:\ONNX\onnxruntime-win-x64-gpu-1.24.4\include  ← ONNX Runtime
├── D:\opencv\opencv\build\include           ← OpenCV
└── imgui/                                   ← Dear ImGui (源码内置)

链接库 (Link):
├── NvInfer_10.lib          ← TensorRT 推理核心
├── NvInfer_10_builder.lib  ← TensorRT Engine 构建
├── cudart.lib              ← CUDA Runtime
├── onnxruntime.lib         ← ONNX Runtime
├── opencv_world.lib        ← OpenCV 全功能
├── d3d11.lib / dxgi.lib    ← DirectX 图形
└── winmm.lib               ← 高精度定时器
```

---

## 编译指南

### 前置条件

| 组件 | 要求 |
|------|------|
| **操作系统** | Windows 10 / 11 (x64) |
| **IDE** | Visual Studio 2022 (v143 工具集) |
| **C++ 标准** | C++20 |
| **GPU** | NVIDIA GPU (支持 CUDA Compute Capability ≥ 6.1) |
| **显卡驱动** | 最新 Game Ready / Studio Driver |

### 安装依赖

1. **CUDA Toolkit 12.8**
   ```
   https://developer.nvidia.com/cuda-downloads
   ```
   默认安装路径：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`

2. **TensorRT 10.16**
   ```
   https://developer.nvidia.com/tensorrt
   ```
   解压到 `E:\TensorRT-10.16.0.72`（或修改 `.vcxproj` 中的路径）

3. **ONNX Runtime GPU 1.24.4**
   ```
   https://github.com/microsoft/onnxruntime/releases/tag/v1.24.4
   选择: onnxruntime-win-x64-gpu-1.24.4.zip
   ```
   解压到 `D:\ONNX\onnxruntime-win-x64-gpu-1.24.4`

4. **OpenCV**
   ```
   https://opencv.org/releases/
   ```
   解压到 `D:\opencv\opencv\build`

### 路径配置

项目使用 `.vcxproj` 中的绝对路径引用上述库。如果需要安装到不同位置，请修改以下属性：

- `<IncludePath>` — 头文件搜索路径
- `<LibraryPath>` — 库文件搜索路径
- `<AdditionalIncludeDirectories>` — 编译时的附加包含目录
- `<AdditionalLibraryDirectories>` — 链接时的附加库目录
- `<AdditionalDependencies>` — 链接的 .lib 文件列表

### 编译步骤

```bash
# 1. 打开 Visual Studio 2022
# 2. 打开项目文件
ScreenCaptureModule.vcxproj

# 3. 选择配置: Release | x64
# 4. 构建 (Ctrl+Shift+B)

# 输出:
# x64\Release\ScreenCaptureModule.exe  (独立程序)
# 或 x64\Release\ScreenCaptureModule.dll  (DLL 配置)
```

---

## 模型准备

### 模型格式

| 格式 | 文件后缀 | 推理引擎 | 说明 |
|------|---------|---------|------|
| TensorRT Engine | `.engine` | TensorRT | 最快。需用 `trtexec` 从 ONNX 转换 |
| ONNX 模型 | `.onnx` | ONNX Runtime | 通用格式，CPU/GPU 均可推理 |

### 导出流程

```bash
# 1. 训练 YOLOv10/v11 模型 → 导出 ONNX
python export.py --weights model.pt --imgsz 320 --opset 12 --include onnx

# 2. ONNX → TensorRT Engine (推荐)
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16 \
        --minShapes=images:1x3x320x320 \
        --optShapes=images:1x3x320x320 \
        --maxShapes=images:1x3x320x320

# 3. 将 model.engine 放在程序可访问的路径
# 默认参考路径: E:\TensorRT-10.16.0.72\bin\model\wa.engine
```

### 支持的 YOLO 版本

| 版本 | `model_way` | 输出格式 |
|------|------------|---------|
| YOLOv10 | `0` | `[1, 300, 6]` — (x1, y1, x2, y2, conf, class) 绝对像素坐标 |
| YOLOv11 | `1` | `[1, 4+nc, anchors]` — 相对坐标 + 目标置信度 + 类别 |

> 程序会**自动检测** Engine / ONNX 输出张量形状，无需手动指定 `model_way`。

---

## 使用说明

### 启动

```bash
# 直接运行
ScreenCaptureModule.exe

# 推荐：管理员权限运行（DXGI 截图可能需要）
右键 → 以管理员身份运行
```

### ImGui 控制面板操作

启动后按配置的快捷键打开/关闭控制面板。界面分为 7 个标签页：

| 标签页 | 功能 |
|--------|------|
| **Monitor** | 实时 FPS、锁定状态、置信度、距离、速度、折线图监控 |
| **PID** | Kp/Ki/Kd 比例积分微分参数、预测权重、输出限幅 |
| **Aim** | 瞄准点偏移、死区、自动开火、追踪参数 |
| **Advanced** | 距离分区限速、贝塞尔曲线、振荡抑制 |
| **FindColor** | HSV 找色参数（辅助模式） |
| **Device** | 推理引擎选择、模型路径、类别 ID、KMBox 连接、功能开关 |
| **Config** | 保存/加载配置到 `aim_config.ini` |

### 基本工作流程

1. **Device 标签** → 选择推理引擎 (TensorRT / ONNX) 和模型路径
2. **Aim 标签** → 设置瞄准点偏移、死区
3. **PID 标签** → 根据手感调节控制参数
4. **启动运行** → 进入游戏，按下瞄准触发键（左键/右键）
5. **Config 标签** → 满意后保存配置

---

## 配置参数

配置文件 `aim_config.ini` 示例（共 60+ 可调参数）：

```ini
# --- PID 控制 ---
kp=0.2                  # 比例增益
ki=0.018                # 积分增益
kd=0.002                # 微分增益
pred_wx=1.2             # X 轴预测权重
pred_wy=0.3             # Y 轴预测权重
pred_zd=25              # 预测零距离
pred_fd=80              # 预测全距
out_max=150             # 最大输出限制
ramp=150                # 初始斜坡时长 (ms)
init_scale=0.75         # 初始 P 缩放比
dz=3                    # 死区 (px)
iclear=12               # 积分清除距离

# --- 滤波 ---
filter_alpha=0.18       # 低通滤波 alpha
filter_nd=80            # 滤波无衰减距离
osc_decay=0.2           # 振荡衰减
osc_nd=70               # 振荡检测距离
osc_sf=4                # 振荡敏感度

# --- 距离分区 ---
near=20                 # 近距离阈值 (px)
mid=45                  # 中距离阈值 (px)
far=90                  # 远距离阈值 (px)
maxN=3                  # 近区最大步长
maxM=12                 # 中区最大步长
maxFF=32                # 远区最大步长
maxF=110                # 超远区最大步长

# --- 贝塞尔曲线 ---
bez_thresh=50           # 触发距离阈值 (px)
bez_dev=0.28            # 轨迹偏移量

# --- 瞄准 ---
y_off=12                # Y 轴偏移
max_lost=6              # 最大丢失帧数
predict_t=0.007         # 预测时间偏移 (s)
vel_alpha=0.35          # 速度 EMA alpha
max_vel=800             # 最大速度 (px/s)
same_sq=2500            # 同一目标判定距离²
conf=0.25               # 置信度阈值
auto_fire=1             # 自动开火 (1=开 0=关)
fire_r=3                # 开火半径 (px)
aim_pct=-30             # 瞄准点高度百分比

# --- 截图 ---
cap_r=112               # 截图半径 (224×224)

# --- 类别 ---
categories=0,5          # 目标类别 ID

# --- 找色 ---
findcolor=0             # 找色开关
```

---

## 项目结构

```
aim/
├── main.cpp                    # 入口点、主循环、线程管理、瞄准逻辑
├── all.h                       # 统一包含头文件
│
├── Detector.h/.cpp             # YOLO 推理类 (TensorRT + ONNX)
│
├── IScreenCapture.h            # 共享接口 (BMP 头、内存池、纹理缓存)
├── ScreenCapture.h/.cpp        # 截图统一门面类
├── DXScreenCapture.h/.cpp      # DXGI Desktop Duplication 截图
├── GDIScreenCapture.h/.cpp     # GDI BitBlt 截图 (兜底)
├── WGCScreenCapture.h/.cpp     # Windows Graphics Capture 截图
│
├── pid.h/.cpp                  # PID 控制器 + 贝塞尔曲线 + 振荡抑制
├── PIDController.h/.cpp        # 高级 PID (Taylor 预测)
│
├── kmboxNet.h/.cpp             # KMBox Net 硬件通信协议
├── exports.h/.cpp              # DLL 导出接口
│
├── findcolor.h                 # HSV 找色辅助
├── HidTable.h                  # USB HID 键盘码表
├── my_enc.h/.cpp               # 加密函数
│
├── ImGuiUI.h                   # ImGui 控制面板 (7 标签页)
├── imgui/                      # Dear ImGui 库 (源码内置)
│   ├── imgui.h/.cpp
│   ├── imgui_draw.cpp
│   ├── imgui_widgets.cpp
│   ├── imgui_tables.cpp
│   ├── imgui_impl_dx11.h/.cpp
│   ├── imgui_impl_win32.h/.cpp
│   └── imstb_*.h
│
├── aim_config.ini              # 配置文件
├── imgui.ini                   # ImGui 窗口状态
├── ScreenCaptureModule.vcxproj # VS2022 项目文件
└── README.md                   # 本文件
```

---

## 许可

本项目仅供**学术研究**与**技术学习**使用。请勿将其用于任何违反游戏服务条款的活动。使用者应自行承担所有责任。

---

<div align="center">
  <sub>Built with ❤️ using C++20 · CUDA · TensorRT · ONNX Runtime · OpenCV · DirectX · ImGui</sub>
</div>
