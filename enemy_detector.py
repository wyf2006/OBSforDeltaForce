"""
YOLO敌人检测模块 - 使用YOLOv8进行实时目标检测
检测FPS游戏中的敌人目标，返回敌人位置坐标

支持：
- 自定义训练的YOLO模型
- YOLOv8预训练模型（检测"person"类别）
- 可选CUDA加速
- 置信度阈值过滤
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class DetectionResult:
    """检测结果数据类"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 边界框
    center: Tuple[int, int]          # (cx, cy) 中心点坐标
    confidence: float                # 置信度 0-1
    class_id: int                    # 类别ID
    class_name: str                  # 类别名称


class EnemyDetector:
    """
    YOLO敌人检测器
    """

    # YOLOv8 COCO数据集中person类的ID
    PERSON_CLASS_ID = 0
    PERSON_CLASS_NAME = "person"

    def __init__(self, model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 device: str = "auto",
                 target_classes: Optional[List[str]] = None):
        """
        初始化YOLO检测器

        Args:
            model_path: YOLO模型路径（None则使用yolov8n.pt预训练模型）
            confidence_threshold: 置信度阈值，低于此值的结果将被过滤
            device: 推理设备 - "auto", "cpu", "cuda", "0"(GPU编号)
            target_classes: 需要检测的目标类别列表，如 ["person", "enemy"]
                           None表示训练模型时定义的默认类别
        """
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes

        # 设备处理：ultralytics的"auto"有子进程CUDA bug，手动解析
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "0"  # 直接用GPU 0，避免子进程可见性问题
                print("[YOLO] 检测到CUDA，使用GPU:0")
            else:
                self.device = "cpu"
                print("[YOLO] 未检测到CUDA，使用CPU")
        else:
            self.device = device

        print(f"[YOLO] 推理设备: {self.device}")

        # 加载模型
        self.model = self._load_model(model_path)

        # 模型类别名列表
        self.model_classes = self._get_model_classes()

        # 性能统计
        self.inference_times = []
        self.last_inference_time = 0

    def _load_model(self, model_path: Optional[str]):
        """加载YOLO模型"""
        from ultralytics import YOLO

        if model_path is None:
            # 使用yolov8n预训练模型（最快的版本，适合实时检测）
            print("[YOLO] 使用YOLOv8n预训练模型")
            print("[YOLO] 如需自定义训练模型，请指定model_path参数")
            try:
                model = YOLO("yolov8n.pt")
            except Exception:
                # 如果下载失败，尝试yolov8s
                print("[YOLO] 尝试使用yolov8s.pt...")
                model = YOLO("yolov8s.pt")
        else:
            print(f"[YOLO] 加载自定义模型: {model_path}")
            model = YOLO(model_path)

        print(f"[YOLO] 模型加载完成，设备: {self.device}")
        return model

    def _get_model_classes(self) -> dict:
        """获取模型的类别名称映射"""
        # YOLOv8的names属性可能在不同版本中有所差异
        try:
            if hasattr(self.model, 'names'):
                return self.model.names
        except Exception:
            pass

        # 默认COCO类别映射
        return {
            0: "person",
        }

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        对一帧画面进行目标检测

        Args:
            frame: BGR格式的图像帧 (numpy数组)

        Returns:
            检测结果列表，按置信度降序排列
        """
        start_time = time.perf_counter()

        # 使用YOLOv8进行推理
        results = self.model(
            frame,
            device=self.device,
            verbose=False,       # 不打印详细信息
            conf=self.confidence_threshold,
            iou=0.45,            # NMS IoU阈值
            max_det=20,          # 最大检测数量（游戏中通常不会超过20个敌人）
        )

        self.last_inference_time = (time.perf_counter() - start_time) * 1000
        self.inference_times.append(self.last_inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)

        # 解析检测结果
        detections = []
        if len(results) > 0:
            result = results[0]  # 第一张图片的结果
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # 获取置信度和类别
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])

                    # 类别过滤
                    class_name = self.model_classes.get(class_id, f"class_{class_id}")
                    if self.target_classes and class_name not in self.target_classes:
                        continue

                    # 计算中心点
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    detection = DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name,
                    )
                    detections.append(detection)

        # 按置信度降序排列
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def get_nearest_enemy(self, detections: List[DetectionResult],
                          frame_center: Tuple[int, int]) -> Optional[DetectionResult]:
        """
        获取离画面中心最近的敌人

        Args:
            detections: 检测结果列表
            frame_center: 画面中心坐标 (cx, cy)

        Returns:
            最近的敌人检测结果，没有则返回None
        """
        if not detections:
            return None

        cx_center, cy_center = frame_center

        def distance_to_center(det: DetectionResult) -> float:
            cx, cy = det.center
            return ((cx - cx_center) ** 2 + (cy - cy_center) ** 2) ** 0.5

        return min(detections, key=distance_to_center)

    def get_largest_enemy(self, detections: List[DetectionResult]) -> Optional[DetectionResult]:
        """
        获取画面中面积最大的敌人（通常表示最近的敌人）

        Args:
            detections: 检测结果列表

        Returns:
            面积最大的敌人检测结果
        """
        if not detections:
            return None

        def bbox_area(det: DetectionResult) -> int:
            x1, y1, x2, y2 = det.bbox
            return (x2 - x1) * (y2 - y1)

        return max(detections, key=bbox_area)

    def get_average_inference_time(self) -> float:
        """获取平均推理时间（毫秒）"""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times)

    def draw_detections(self, frame: np.ndarray,
                        detections: List[DetectionResult],
                        highlight_target: Optional[DetectionResult] = None,
                        frame_center: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        在画面上绘制检测结果（调试/可视化用）

        Args:
            frame: 原始帧
            detections: 检测结果
            highlight_target: 高亮显示的目标
            frame_center: 画面中心点

        Returns:
            绘制后的帧
        """
        display = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx, cy = det.center

            # 边界框颜色（高亮目标用红色，其他用绿色）
            if highlight_target and det is highlight_target:
                color = (0, 0, 255)  # 红色 (BGR)
                thickness = 3
            else:
                color = (0, 255, 0)  # 绿色 (BGR)
                thickness = 2

            # 绘制边界框
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

            # 绘制中心点
            cv2.circle(display, (cx, cy), 5, color, -1)

            # 绘制标签
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)
            cv2.putText(display, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 绘制画面中心十字线
        if frame_center:
            fcx, fcy = frame_center
            cv2.drawMarker(display, (fcx, fcy), (255, 0, 0),
                           cv2.MARKER_CROSS, 20, 2)

        # 如果高亮目标，绘制连线
        if highlight_target and frame_center:
            fcx, fcy = frame_center
            tcx, tcy = highlight_target.center
            cv2.line(display, (fcx, fcy), (tcx, tcy), (0, 0, 255), 1)
            cv2.circle(display, (tcx, tcy), 10, (0, 0, 255), 2)

        # 绘制FPS和推理时间信息
        info_lines = [
            f"Detections: {len(detections)}",
            f"Inference: {self.last_inference_time:.1f}ms",
        ]
        y_offset = 30
        for line in info_lines:
            cv2.putText(display, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

        return display
