# FPS YOLO 瞄准辅助系统

注意，这完全是练习用的半成品，现在还没有实际作用！！！

## 📋 系统架构

```
OBS捕获画面 → Python读取画面 → YOLO检测敌人 → 计算鼠标偏移 → 控制鼠标瞄准
```

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ OBS捕获  │───▶│ screen_capture│───▶│enemy_detector│───▶│   mouse_     │
│ 游戏画面  │    │   .py        │    │   .py        │    │ controller.py│
└──────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                         │                    │
                                    YOLOv8模型           移动鼠标到目标
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置OBS虚拟摄像头（推荐）

1. 打开OBS Studio
2. 添加游戏窗口/显示器作为源
3. 点击 `工具` → `虚拟摄像头` → `启动`
4. 确认OBS虚拟摄像头已启动

### 3. 调整配置

编辑 `main.py` 中的 `Config` 类：

```python
# 捕获模式: "virtual_camera" 或 "screen"
CAPTURE_MODE = "screen"        # 如果用OBS虚拟摄像头改为 "virtual_camera"

# 激活键 (按住激活瞄准)
ACTIVATION_KEY = "shift"       # 可选: "ctrl", "alt", "x1", "x2"

# 灵敏度 - 根据你的游戏灵敏度调整
SENSITIVITY = 0.8

# 置信度阈值 (推荐0.3-0.5)
CONFIDENCE_THRESHOLD = 0.35
```

### 4. 运行

```bash
python main.py
```

## ⌨️ 操作说明

| 按键 | 功能 |
|------|------|
| 按`F8` | 激活瞄准（默认） |
| `F9` 或 `Q` | 退出程序 |
| `1` | 切换策略：瞄准最近敌人 |
| `2` | 切换策略：瞄准最大敌人 |
| `3` | 切换策略：瞄准最高置信度敌人 |

## ⚙️ 配置项说明

### 捕获模式

```python
# OBS虚拟摄像头模式（OBS输出已处理好画面）
CAPTURE_MODE = "virtual_camera"
CAMERA_ID = 0  # 虚拟摄像头设备ID

# 屏幕捕获模式（直接用Python截屏）
CAPTURE_MODE = "screen"
MONITOR_INDEX = 1  # 1=主显示器
```

### 瞄准参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `SENSITIVITY` | 灵敏度，值越大鼠标移动幅度越大 | 0.5-1.5 |
| `SMOOTHING` | 平滑度(0-1)，越小越平滑 | 0.4-0.6 |
| `DEADZONE_RADIUS` | 死区半径(像素) | 2-5 |
| `HEADSHOT_OFFSET` | 爆头偏移(0=头顶, 1=脚底) | 0.2-0.3 |

### 激活键

```python
ACTIVATION_KEY = "shift"       # 按住Shift激活
# 其他选项:
# "ctrl", "alt", "caps_lock"
# "x1" (鼠标侧键前进), "x2" (鼠标侧键后退)
# "mouse_middle" (鼠标中键)

ALWAYS_ACTIVE = False  # 设为True始终激活（不推荐）
```

## 🎯 YOLO模型

### 方案1: 使用预训练模型（快速开始）

默认使用 `yolov8n.pt` 预训练模型，检测 `person` 类别。适合大多数FPS游戏（敌人是人形角色）。

### 方案2: 自定义训练（更准确）

使用 `train_yolo.py` 训练自己的模型：

```bash
python train_yolo.py
```

自定义训练步骤：
1. 准备数据集（游戏截图 + 标注）
2. 按照YOLO格式组织数据
3. 运行训练脚本

## ⚠️ 注意事项

1. **仅供学习研究**：本项目仅用于教学目的，展示YOLO目标检测和鼠标控制技术
2. **公平游戏**：在多人游戏中请遵守游戏规则
3. **反作弊检测**：部分游戏有反作弊系统，请注意风险
4. **性能**：建议使用GPU（CUDA）进行推理，CPU推理可能较慢
5. **OBS延迟**：OBS虚拟摄像头有一定延迟，屏幕捕获模式延迟更低

## 📁 项目文件

```
DELTAFORCEOBS/
├── main.py              # 主程序入口
├── screen_capture.py    # 画面捕获模块
├── enemy_detector.py    # YOLO检测模块
├── mouse_controller.py  # 鼠标控制模块
├── train_yolo.py        # YOLO训练脚本
├── requirements.txt     # 依赖列表
└── README.md            # 本文件
```
