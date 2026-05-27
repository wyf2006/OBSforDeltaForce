"""
OBS画面捕获模块 - 通过OBS虚拟摄像头或屏幕捕获获取游戏画面
支持两种模式：
1. OBS虚拟摄像头模式（推荐：OBS输出设置为虚拟摄像头）
2. 屏幕捕获模式（直接捕获显示器画面）
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import time


class ScreenCapture:
    """
    画面捕获类，支持OBS虚拟摄像头和屏幕截图两种方式
    """

    def __init__(self, mode: str = "virtual_camera", camera_id: int = 0,
                 monitor_index: int = 1, target_fps: int = 60):
        """
        初始化画面捕获

        Args:
            mode: 捕获模式 - "virtual_camera" 或 "screen"
            camera_id: 虚拟摄像头的设备ID（OBS虚拟摄像头默认通常是0或1）
            monitor_index: 屏幕捕获时的显示器编号
            target_fps: 目标帧率
        """
        self.mode = mode
        self.camera_id = camera_id
        self.monitor_index = monitor_index
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.sct = None
        self.monitor = None
        self.last_frame_time = 0
        self.is_running = False

    def start(self) -> bool:
        """
        启动画面捕获

        Returns:
            是否成功启动
        """
        if self.mode == "virtual_camera":
            return self._start_virtual_camera()
        elif self.mode == "screen":
            return self._start_screen_capture()
        else:
            raise ValueError(f"不支持的捕获模式: {self.mode}")

    def _start_virtual_camera(self) -> bool:
        """启动OBS虚拟摄像头捕获"""
        # OBS虚拟摄像头通常使用DShow后端
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            # 尝试使用默认后端
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"[错误] 无法打开摄像头设备 {self.camera_id}")
                print("[提示] 请确保OBS已启动虚拟摄像头功能")
                print("[提示] 可用摄像头列表:")
                self._list_cameras()
                return False

        # 设置高分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"[虚拟摄像头] 已连接 - 分辨率: {actual_w:.0f}x{actual_h:.0f}, FPS: {actual_fps:.0f}")
        self.is_running = True
        return True

    def _start_screen_capture(self) -> bool:
        """启动屏幕捕获（使用mss库，性能优于pyautogui）"""
        try:
            import mss
            self.sct = mss.mss()
            monitors = self.sct.monitors

            if self.monitor_index >= len(monitors):
                print(f"[错误] 显示器索引 {self.monitor_index} 超出范围")
                print(f"[提示] 可用显示器: {len(monitors)} 个")
                for i, m in enumerate(monitors):
                    print(f"  显示器 {i}: {m}")
                return False

            self.monitor = monitors[self.monitor_index]
            print(f"[屏幕捕获] 已就绪 - 显示器 {self.monitor_index}: {self.monitor}")
            self.is_running = True
            return True
        except ImportError:
            print("[错误] 请安装mss库: pip install mss")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        捕获一帧画面

        Returns:
            BGR格式的numpy数组，失败返回None
        """
        if self.mode == "virtual_camera":
            return self._capture_from_camera()
        elif self.mode == "screen":
            return self._capture_from_screen()
        return None

    def _capture_from_camera(self) -> Optional[np.ndarray]:
        """从虚拟摄像头捕获帧"""
        if self.cap is None:
            return None

        # 清空缓冲区以获取最新帧
        for _ in range(2):
            self.cap.grab()

        ret, frame = self.cap.read()
        if not ret:
            print("[警告] 读取摄像头帧失败")
            return None

        return frame

    def _capture_from_screen(self) -> Optional[np.ndarray]:
        """从屏幕捕获帧"""
        if self.sct is None or self.monitor is None:
            return None

        try:
            # 捕获屏幕区域
            screenshot = self.sct.grab(self.monitor)
            # 转换为numpy数组 (BGRA -> BGR)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            print(f"[警告] 屏幕捕获异常: {e}")
            return None

    def get_frame_size(self) -> Tuple[int, int]:
        """获取捕获画面的尺寸 (宽, 高)"""
        frame = self.capture_frame()
        if frame is not None:
            return (frame.shape[1], frame.shape[0])
        return (1920, 1080)  # 默认值

    def stop(self):
        """停止捕获"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.sct is not None:
            self.sct.close()
            self.sct = None
        print("[捕获] 已停止")

    @staticmethod
    def _list_cameras():
        """列出所有可用的摄像头设备"""
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"  摄像头 {i}: {w:.0f}x{h:.0f}")
                cap.release()

    def show_preview(self, frame: np.ndarray, window_name: str = "OBS Preview"):
        """
        显示预览画面（调试用）

        Args:
            frame: 要显示的帧
            window_name: 窗口名称
        """
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
