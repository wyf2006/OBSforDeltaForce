"""
YOLO自定义训练脚本 - 用于训练FPS游戏敌人检测模型

使用方法:
1. 准备数据集（图片 + YOLO格式标注）
2. 修改下方配置
3. 运行: python train_yolo.py

数据集目录结构:
dataset/
├── images/
│   ├── train/          # 训练图片
│   └── val/            # 验证图片
├── labels/
│   ├── train/          # YOLO格式标注 (.txt)
│   └── val/            # YOLO格式标注 (.txt)
└── data.yaml           # 数据集配置文件
"""

import os
from ultralytics import YOLO


# ========================
# 训练配置
# ========================
class TrainConfig:
    # 数据集配置文件路径
    DATA_YAML = "dataset/data.yaml"

    # 预训练模型（在此基础上微调）
    # "yolov8n.pt" - 最快, "yolov8s.pt" - 平衡, "yolov8m.pt" - 更准
    PRETRAINED_MODEL = "yolov8n.pt"

    # 训练轮数
    EPOCHS = 100

    # 图片尺寸
    IMGSZ = 640

    # 批次大小
    BATCH = 16

    # 设备: "auto", "0"(GPU0), "cpu"
    DEVICE = "auto"

    # 工作进程数
    WORKERS = 4

    # 学习率
    LR0 = 0.01

    # 输出目录
    PROJECT = "runs/detect"
    NAME = "fps_enemy_detector"

    # 是否使用预训练权重
    USE_PRETRAINED = True

    # 数据增强（FPS游戏场景建议开启）
    # mosaic: 马赛克增强, mixup: 混合增强
    MOSAIC = 1.0       # 1.0=100%使用, 0.0=禁用
    MIXUP = 0.0        # 建议关闭mixup（游戏场景中效果一般）
    HFLIP = 0.5        # 水平翻转概率
    SCALE = 0.5        # 缩放增强


def create_data_yaml(dataset_dir: str, classes: list):
    """
    创建YOLO数据集配置文件

    Args:
        dataset_dir: 数据集根目录
        classes: 类别名称列表
    """
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    # 使用绝对路径（Linux需要正斜杠）
    abs_path = os.path.abspath(dataset_dir).replace("\\", "/")

    content = f"""# FPS游戏敌人检测数据集配置

# 数据集路径
path: {abs_path}
train: images/train
val: images/val

# 类别
nc: {len(classes)}
names: {classes}
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[数据集] 配置文件已创建: {yaml_path}")
    print(f"[数据集] 类别数: {len(classes)}")
    print(f"[数据集] 类别: {classes}")

    return yaml_path


def train():
    """训练YOLO模型"""
    config = TrainConfig()

    print("=" * 50)
    print("  YOLO FPS敌人检测 - 训练脚本")
    print("=" * 50)

    # 检查数据集配置
    if not os.path.exists(config.DATA_YAML):
        print(f"\n[提示] 数据集配置文件不存在: {config.DATA_YAML}")
        print("[提示] 请先准备数据集，或使用以下代码创建示例配置:\n")
        print("  from train_yolo import create_data_yaml")
        print("  create_data_yaml('dataset/', ['enemy', 'ally', 'head'])")
        print()
        return

    # 加载模型
    print(f"\n[模型] 加载基础模型: {config.PRETRAINED_MODEL}")
    model = YOLO(config.PRETRAINED_MODEL)

    # 开始训练
    print(f"\n[训练] 开始训练...")
    print(f"  数据集: {config.DATA_YAML}")
    print(f"  轮数: {config.EPOCHS}")
    print(f"  图片尺寸: {config.IMGSZ}")
    print(f"  批次: {config.BATCH}")
    print(f"  设备: {config.DEVICE}")

    results = model.train(
        data=config.DATA_YAML,
        epochs=config.EPOCHS,
        imgsz=config.IMGSZ,
        batch=config.BATCH,
        device=config.DEVICE,
        workers=config.WORKERS,
        lr0=config.LR0,
        project=config.PROJECT,
        name=config.NAME,

        # 数据增强
        mosaic=config.MOSAIC,
        mixup=config.MIXUP,
        hflip=config.HFLIP,
        scale=config.SCALE,

        # 预训练
        pretrained=config.USE_PRETRAINED,

        # 其他
        save=True,         # 保存模型
        save_period=10,    # 每10轮保存一次
        val=True,          # 验证
        plots=True,        # 生成图表
    )

    # 评估
    print(f"\n[评估] 在验证集上评估...")
    metrics = model.val()

    print(f"\n[完成] 训练完成！")
    print(f"  最佳模型: {config.PROJECT}/{config.NAME}/weights/best.pt")
    print(f"  最终模型: {config.PROJECT}/{config.NAME}/weights/last.pt")

    return model, results


def export_model(model_path: str, format: str = "onnx"):
    """
    导出模型到其他格式（用于部署/加速）

    Args:
        model_path: 训练好的模型路径
        format: 导出格式 - "onnx", "engine"(TensorRT), "openvino"等
    """
    model = YOLO(model_path)
    model.export(format=format)
    print(f"[导出] 模型已导出为 {format} 格式")


def test_model(model_path: str, image_path: str, conf: float = 0.5):
    """
    测试训练好的模型

    Args:
        model_path: 模型路径
        image_path: 测试图片路径
        conf: 置信度阈值
    """
    model = YOLO(model_path)
    results = model(image_path, conf=conf, save=True, show=True)

    # 打印检测结果
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf_val = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                print(f"  检测到: {cls_name} (置信度: {conf_val:.2f}) - 位置: {bbox}")


if __name__ == "__main__":
    train()
