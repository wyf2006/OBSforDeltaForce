"""
鼠标瞄准控制模块 - 计算瞄准偏移量并控制鼠标移动

核心功能：
1. 根据敌人坐标和画面中心计算鼠标偏移量
2. 考虑游戏灵敏度进行坐标转换
3. 平滑移动鼠标实现类人的瞄准行为
4. 支持多种瞄准策略（最近敌人/最大敌人/头部优先）
"""

import math
import time
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class AimStrategy(Enum):
    """瞄准策略"""
    NEAREST = "nearest"          # 瞄准离准星最近的敌人
    LARGEST = "largest"           # 瞄准面积最大的敌人
    HIGHEST_CONFIDENCE = "highest"  # 瞄准置信度最高的敌人


@dataclass
class AimConfig:
    """瞄准配置"""
    # 灵敏度倍数 - 调整鼠标移动幅度。值越大移动幅度越大
    sensitivity: float = 1.0

    # 平滑因子 - 鼠标移动的平滑程度 (0-1)，越小越平滑但反应越慢
    smoothing: float = 0.6

    # 死区 - 目标在画面中心的这个像素范围内不移动鼠标
    deadzone_radius: float = 3.0

    # 最小移动距离 - 低于此像素的偏移不做移动
    min_move_distance: float = 1.0

    # 最大单次移动距离 - 防止异常大幅度移动
    max_move_distance: float = 300.0

    # 瞄准策略
    strategy: AimStrategy = AimStrategy.NEAREST

    # 瞄准偏移 - 正值表示瞄准敌人上方（爆头偏移），单位: 边界框高度的比例
    # 例如: 0.3 表示瞄准边界框顶部下方30%的位置
    headshot_offset: float = 0.25

    # Y轴反转（某些游戏需要）
    invert_y: bool = False

    # 移动曲线 - "linear" 线性, "ease_out" 缓出(开始时快,接近目标时慢)
    movement_curve: str = "ease_out"


class MouseController:
    """
    鼠标控制器 - 计算偏移量并控制鼠标
    """

    def __init__(self, config: Optional[AimConfig] = None):
        self.config = config or AimConfig()

        # 平滑用的历史偏移量（用于EMA平滑）
        self._prev_offset_x = 0.0
        self._prev_offset_y = 0.0

        # 统计
        self.last_offset: Tuple[float, float] = (0, 0)
        self.is_aiming = False

    def calculate_offset(self, target_pos: Tuple[int, int],
                         frame_center: Tuple[int, int],
                         bbox: Optional[Tuple[int, int, int, int]] = None,
                         frame_size: Optional[Tuple[int, int]] = None
                         ) -> Tuple[float, float]:
        """
        计算鼠标需要移动的偏移量

        Args:
            target_pos: 目标在画面中的坐标 (x, y)
            frame_center: 画面中心坐标 (cx, cy) - 即准星位置
            bbox: 目标的边界框 (x1, y1, x2, y2)，用于爆头偏移计算
            frame_size: 画面尺寸 (w, h)，用于边界检查

        Returns:
            鼠标偏移量 (dx, dy)，单位：像素移动量
        """
        tx, ty = target_pos
        fcx, fcy = frame_center

        # 如果提供了边界框，应用爆头偏移
        if bbox is not None:
            _, y1, _, y2 = bbox
            bbox_height = y2 - y1
            # 瞄准边界框上方一定比例（通常头部在身体上方）
            ty = y1 + int(bbox_height * self.config.headshot_offset)

        # 计算原始偏移量
        raw_dx = tx - fcx
        raw_dy = ty - fcy

        # 死区检查
        distance = math.sqrt(raw_dx ** 2 + raw_dy ** 2)
        if distance < self.config.deadzone_radius:
            return (0.0, 0.0)

        # 限制最大移动距离
        if distance > self.config.max_move_distance:
            scale = self.config.max_move_distance / distance
            raw_dx *= scale
            raw_dy *= scale

        # 应用移动曲线
        dx, dy = self._apply_movement_curve(raw_dx, raw_dy, distance)

        # 应用灵敏度
        dx *= self.config.sensitivity
        dy *= self.config.sensitivity

        # Y轴反转
        if self.config.invert_y:
            dy = -dy

        # 最小移动检查
        if abs(dx) < self.config.min_move_distance and abs(dy) < self.config.min_move_distance:
            return (0.0, 0.0)

        return (dx, dy)

    def smooth_offset(self, dx: float, dy: float) -> Tuple[float, float]:
        """
        对偏移量进行平滑处理（EMA指数移动平均）

        Args:
            dx: 当前帧计算的X偏移量
            dy: 当前帧计算的Y偏移量

        Returns:
            平滑后的偏移量
        """
        s = self.config.smoothing
        smooth_dx = self._prev_offset_x * (1 - s) + dx * s
        smooth_dy = self._prev_offset_y * (1 - s) + dy * s

        self._prev_offset_x = smooth_dx
        self._prev_offset_y = smooth_dy

        return (smooth_dx, smooth_dy)

    def _apply_movement_curve(self, dx: float, dy: float,
                               distance: float) -> Tuple[float, float]:
        """
        应用移动曲线，使瞄准行为更自然

        - linear: 线性移动
        - ease_out: 先快后慢（远距离快速移动，近距离微调）
        """
        if self.config.movement_curve == "linear":
            return (dx, dy)
        elif self.config.movement_curve == "ease_out":
            # 使用平方根曲线：远距离时比例更大
            # normalized_distance范围[0, 1]
            norm_dist = min(distance / self.config.max_move_distance, 1.0)
            # ease_out: 1 - (1-t)^2
            factor = 1.0 - (1.0 - norm_dist) ** 2
            # 将factor映射到实际偏移比例
            # 远距离时factor接近1，近距离时factor较小，但不会为0
            scale = 0.3 + factor * 0.7  # [0.3, 1.0]
            return (dx * scale, dy * scale)

        return (dx, dy)

    def move_mouse_absolute(self, x: int, y: int):
        """
        绝对定位：直接把系统光标跳到指定屏幕坐标

        Args:
            x: 主屏X坐标
            y: 主屏Y坐标
        """
        self.is_aiming = True
        self.last_offset = (float(x), float(y))

        import ctypes
        # SetCursorPos(x, y) - Win32绝对定位
        ctypes.windll.user32.SetCursorPos(x, y)

    def move_mouse(self, dx: float, dy: float,
                   use_win32: bool = True):
        """
        移动鼠标

        Args:
            dx: X轴相对移动量
            dy: Y轴相对移动量
            use_win32: True使用Win32 SendInput（最可靠），False回退pyautogui
        """
        if abs(dx) < self.config.min_move_distance and abs(dy) < self.config.min_move_distance:
            self.is_aiming = False
            return

        self.is_aiming = True
        self.last_offset = (dx, dy)

        dx_int = int(round(dx))
        dy_int = int(round(dy))

        if use_win32:
            # Win32 SendInput 相对鼠标移动，在所有DirectX游戏中都有效
            self._win32_mouse_move(dx_int, dy_int)
        else:
            try:
                import pydirectinput
                pydirectinput.FAILSAFE = False
                pydirectinput.moveRel(dx_int, dy_int, relative=True)
            except (ImportError, TypeError):
                import pyautogui
                pyautogui.FAILSAFE = False
                pyautogui.moveRel(dx_int, dy_int, duration=0)

    @staticmethod
    def _win32_mouse_move(dx: int, dy: int):
        """通过Win32 SendInput发送鼠标相对移动事件"""
        import ctypes
        from ctypes import wintypes

        # 定义 INPUT 结构
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]

        class INPUT(ctypes.Structure):
            class _INPUT(ctypes.Union):
                _fields_ = [("mi", MOUSEINPUT)]
            _anonymous_ = ("_input",)
            _fields_ = [
                ("type", wintypes.DWORD),
                ("_input", _INPUT),
            ]

        # MOUSEEVENTF_MOVE = 0x0001 相对移动
        MOUSEEVENTF_MOVE = 0x0001
        INPUT_MOUSE = 0

        # 发送输入事件
        extra = ctypes.c_ulong(0)
        ii = INPUT()
        ii.type = INPUT_MOUSE
        ii.mi.dx = dx
        ii.mi.dy = dy
        ii.mi.mouseData = 0
        ii.mi.dwFlags = MOUSEEVENTF_MOVE
        ii.mi.time = 0
        ii.mi.dwExtraInfo = ctypes.pointer(extra)

        ctypes.windll.user32.SendInput(1, ctypes.byref(ii), ctypes.sizeof(ii))

    def reset_smoothing(self):
        """重置平滑状态（切换目标时调用）"""
        self._prev_offset_x = 0.0
        self._prev_offset_y = 0.0
        self.is_aiming = False

    def select_target_by_strategy(self, detections, frame_center):
        """
        根据瞄准策略选择目标

        Args:
            detections: 检测结果列表 (DetectionResult对象列表)
            frame_center: 画面中心坐标

        Returns:
            选中的DetectionResult或None
        """
        if not detections:
            return None

        strategy = self.config.strategy

        if strategy == AimStrategy.NEAREST:
            return self._select_nearest(detections, frame_center)
        elif strategy == AimStrategy.LARGEST:
            return self._select_largest(detections)
        elif strategy == AimStrategy.HIGHEST_CONFIDENCE:
            return self._select_highest_confidence(detections)
        else:
            return self._select_nearest(detections, frame_center)

    def _select_nearest(self, detections, frame_center):
        """选择离准星最近的敌人"""
        fcx, fcy = frame_center
        nearest = None
        min_dist = float('inf')

        for det in detections:
            cx, cy = det.center
            dist = ((cx - fcx) ** 2 + (cy - fcy) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = det

        return nearest

    def _select_largest(self, detections):
        """选择边界框面积最大的敌人"""
        largest = None
        max_area = 0

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest = det

        return largest

    def _select_highest_confidence(self, detections):
        """选择置信度最高的敌人"""
        return max(detections, key=lambda d: d.confidence)
