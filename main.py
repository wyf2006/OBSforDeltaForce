"""
FPS游戏YOLO瞄准辅助 - 主程序

工作流程:
1. OBS捕获游戏画面（虚拟摄像头或屏幕捕获）
2. YOLO实时检测画面中的敌人
3. 根据策略选择目标（最近/最大/最高置信度）
4. 计算鼠标偏移量并平滑移动

操作说明:
- 运行后默认处于"检测-预览"模式（不控制鼠标）
- 按住鼠标侧键(X1/X2) 或 指定热键 激活瞄准
- 支持多种瞄准策略切换
"""

import cv2
import time
import sys
import signal
from typing import Optional

from screen_capture import ScreenCapture
from enemy_detector import EnemyDetector
from mouse_controller import MouseController, AimConfig, AimStrategy


# ========================
# 配置区 - 根据需要修改
# ========================
class Config:
    """全局配置"""

    # --- 捕获设置 ---
    # "virtual_camera": OBS虚拟摄像头（你选的这个）
    # "screen": 直接屏幕捕获
    CAPTURE_MODE = "virtual_camera"

    # 虚拟摄像头设备ID
    # 已检测到电脑摄像头是 0，"OBS Virtual Camera" 是 1
    CAMERA_ID = 1

    # 屏幕捕获的显示器编号（仅CAPTURE_MODE="screen"时生效）
    MONITOR_INDEX = 1

    # --- YOLO设置 ---
    # 模型路径，None使用yolov8n.pt
    MODEL_PATH = None  # 例如: "runs/detect/train/weights/best.pt"

    # 置信度阈值 (0-1)，三角洲行动人物模型建议0.3
    CONFIDENCE_THRESHOLD = 0.3

    # 推理设备: "0"(GPU0), "cuda", "cpu"
    # ultralytics的"auto"有子进程CUDA可见性bug，RTX4070直接用"0"
    DEVICE = "0"

    # 要检测的目标类别
    TARGET_CLASSES = ["person"]

    # --- 瞄准设置 ---
    # 灵敏度 - 值越大鼠标移动越快
    # 三角洲行动建议从0.6开始，根据手感调整
    SENSITIVITY = 0.6

    # 平滑度 (0-1)，越小越平滑但响应越慢
    SMOOTHING = 0.4

    # 死区半径（像素），准星在目标此范围内不移动
    DEADZONE_RADIUS = 5.0

    # 爆头偏移（边界框高度的比例，从上往下）
    # 0.25 大约是头部位置
    HEADSHOT_OFFSET = 0.25

    # 移动曲线: "linear" 线性, "ease_out" 缓出
    MOVEMENT_CURVE = "ease_out"

    # 瞄准策略:
    # "nearest" - 瞄准离准星最近的
    # "largest" - 瞄准面积最大的
    # "highest" - 瞄准置信度最高的
    AIM_STRATEGY = "nearest"

    # --- 激活方式 ---
    # 按一下F8开启/关闭自动瞄准
    ACTIVATION_KEY = "f8"

    # 始终激活模式(不按任何键就生效)，注意安全！
    ALWAYS_ACTIVE = False

    # --- 显示设置 ---
    # 是否显示预览窗口（调试时开启，正式用可以关掉减少卡顿）
    SHOW_PREVIEW = True

    # 预览窗口缩放比例（4K屏幕建议0.45-0.55让窗口不占满整个屏）
    PREVIEW_SCALE = 1

    # 是否绘制检测框
    DRAW_DETECTIONS = True

    # --- 性能设置 ---
    # 目标帧率
    TARGET_FPS = 60

    # YOLO每隔N帧推理一次（1=每帧推理）
    INFERENCE_EVERY_N_FRAMES = 1


class AimAssist:
    """瞄准辅助主类"""

    def __init__(self):
        self.config = Config()

        # 初始化各模块
        print("=" * 50)
        print("  FPS YOLO 瞄准辅助系统")
        print("=" * 50)

        # 1. 画面捕获
        print("\n[1/3] 初始化画面捕获...")
        self.capture = ScreenCapture(
            mode=self.config.CAPTURE_MODE,
            camera_id=self.config.CAMERA_ID,
            monitor_index=self.config.MONITOR_INDEX,
            target_fps=self.config.TARGET_FPS
        )
        if not self.capture.start():
            print("[错误] 画面捕获启动失败，程序退出")
            sys.exit(1)

        # 获取实际画面尺寸，并验证OBS画面是否正常
        print("\n[验证] 检测OBS输出画面...")
        frame = self.capture.capture_frame()
        if frame is not None:
            self.frame_width = frame.shape[1]
            self.frame_height = frame.shape[0]

            # 检查画面是否有效（非纯黑/纯白画面）
            mean_brightness = frame.mean()
            std_brightness = frame.std()

            print(f"  ✅ OBS画面已捕获！")
            print(f"  分辨率: {self.frame_width} x {self.frame_height}")
            print(f"  平均亮度: {mean_brightness:.1f}")
            print(f"  画面变化度: {std_brightness:.1f}")

            if std_brightness < 5.0:
                print(f"  ⚠️  警告：画面变化度极低({std_brightness:.1f})，可能OBS输出的是静止/黑屏画面！")
                print(f"  请检查OBS是否正确捕获了游戏画面，以及虚拟摄像头是否已启动。")
            elif mean_brightness < 10.0:
                print(f"  ⚠️  警告：画面平均亮度极低({mean_brightness:.1f})，可能画面过暗或黑屏！")
                print(f"  请检查OBS源是否正常工作。")
            else:
                print(f"  ℹ️  画面信号正常，可以开始检测。")
        else:
            self.frame_width = 1920
            self.frame_height = 1080
            print(f"  ❌ 无法从OBS读取画面！")
            print(f"  请确认：")
            print(f"    1. OBS → 工具 → 虚拟摄像头 → 已启动")
            print(f"    2. OBS中有画面源（游戏捕获/显示器捕获）")
            print(f"    3. 摄像头ID正确（当前: {self.config.CAMERA_ID}）")
            print(f"  程序将继续运行，但可能检测不到任何目标。")

        # 2. YOLO检测器
        print("\n[2/3] 初始化YOLO检测器...")
        self.detector = EnemyDetector(
            model_path=self.config.MODEL_PATH,
            confidence_threshold=self.config.CONFIDENCE_THRESHOLD,
            device=self.config.DEVICE,
            target_classes=self.config.TARGET_CLASSES
        )

        # 3. 鼠标控制器
        print("\n[3/3] 初始化鼠标控制器...")
        aim_config = AimConfig(
            sensitivity=self.config.SENSITIVITY,
            smoothing=self.config.SMOOTHING,
            deadzone_radius=self.config.DEADZONE_RADIUS,
            headshot_offset=self.config.HEADSHOT_OFFSET,
            movement_curve=self.config.MOVEMENT_CURVE,
            strategy=AimStrategy(self.config.AIM_STRATEGY) if self.config.AIM_STRATEGY in [e.value for e in AimStrategy] else AimStrategy.NEAREST,
        )
        self.mouse = MouseController(config=aim_config)

        # 状态变量
        self.is_running = False
        self.frame_count = 0
        self.fps = 0
        self.fps_timer = time.time()
        self.current_target = None
        self.aim_enabled = False       # 瞄准开关状态
        self.prev_key_state = False    # 上一帧的按键状态

        # 设置退出信号处理
        signal.signal(signal.SIGINT, self._signal_handler)

        print("\n" + "=" * 50)
        print("  初始化完成！")
        print("=" * 50)
        self._print_controls()

    def _print_controls(self):
        """打印操作说明"""
        print(f"\n操作说明:")
        print(f"  [F8] 按一下开启/关闭自动瞄准")
        print(f"  [F9] 退出程序")
        print(f"  瞄准策略: {self.config.AIM_STRATEGY}")
        print(f"  [1/2/3] 切换瞄准策略")
        if self.config.SHOW_PREVIEW:
            print(f"  预览窗口: 已开启")
        print()

    def _signal_handler(self, sig, frame):
        """处理Ctrl+C信号"""
        print("\n[退出] 收到退出信号...")
        self.stop()

    def is_switch_pressed(self) -> bool:
        """检测F8是否按下（使用keyboard的hook事件监听，防抖）"""
        try:
            import keyboard
            return keyboard.is_pressed("f8")
        except ImportError:
            return False

    def _is_exit_key_pressed(self) -> bool:
        """检测 F9 是否被按下（退出程序）"""
        try:
            import keyboard
            if keyboard.is_pressed("f9"):
                return True
            # 也检测Ctrl+C信号触发的退出（兼容）
            if keyboard.is_pressed("ctrl") and keyboard.is_pressed("c"):
                return True
            return False
        except ImportError:
            return False

    def _get_debug_key_info(self) -> str:
        """调试：返回当前按下的所有键"""
        try:
            import keyboard
            pressed = []
            # 检测常见可能导致退出或异常的键
            check_keys = ["f8", "f9", "f10", "f11", "f12", "esc", "q", "ctrl", "alt", "shift", "caps lock",
                          "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                          "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                          "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
            for k in check_keys:
                if keyboard.is_pressed(k):
                    pressed.append(k)
            return "+".join(pressed) if pressed else "无"
        except ImportError:
            return "keyboard库不可用"

    def run(self):
        """主循环"""
        self.is_running = True
        print("[运行] 开始主循环...\n")

        last_time = time.time()

        try:
            while self.is_running:
                loop_start = time.time()

                # 1. 捕获画面
                frame = self.capture.capture_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                self.frame_count += 1

                # 2. YOLO检测（可选间隔推理以提升性能）
                should_infer = (self.frame_count % self.config.INFERENCE_EVERY_N_FRAMES == 0)

                if should_infer:
                    detections = self.detector.detect(frame)
                else:
                    detections = []  # 使用上一帧的检测结果

                # 3. 计算画面中心
                frame_center = (self.frame_width // 2, self.frame_height // 2)

                # 4. 检查激活状态 (切换开关逻辑 - 上升沿检测，防抖)
                f8_now = self.is_switch_pressed()
                if f8_now and not self.prev_key_state:
                    self.aim_enabled = not self.aim_enabled
                    state_str = "开启" if self.aim_enabled else "关闭"
                    print(f"\n[状态] 自动瞄准已{state_str}")
                self.prev_key_state = f8_now

                # F9 退出程序
                if self._is_exit_key_pressed():
                    import keyboard
                    debug_keys = self._get_debug_key_info()
                    print(f"\n[退出] 原因: F9被按下 | 当前按键: [{debug_keys}]")
                    break

                activated = self.aim_enabled or self.config.ALWAYS_ACTIVE

                if activated and detections:
                    # 选择目标
                    target = self.mouse.select_target_by_strategy(detections, frame_center)

                    if target is not None:
                        self.current_target = target

                        # 计算当前帧与屏幕中心的直接像素偏移（每次独立计算，不累积）
                        raw_dx = target.center[0] - frame_center[0]
                        raw_dy = target.center[1] - frame_center[1]

                        # 应用爆头偏移
                        if target.bbox is not None:
                            _, y1, _, y2 = target.bbox
                            bbox_height = y2 - y1
                            head_y = y1 + int(bbox_height * self.mouse.config.headshot_offset)
                            raw_dy = head_y - frame_center[1]

                        # 应用灵敏度（直接缩放，不累积）
                        dx = raw_dx * self.mouse.config.sensitivity
                        dy = raw_dy * self.mouse.config.sensitivity

                        # 动态最大移动距离：根据OBS分辨率智能限制，防止鼠标飞出屏幕
                        # OBS画面和你的4K屏幕比例不同，需要限制单次移动不超过画面宽度的15%
                        max_single_move = min(self.frame_width, self.frame_height) * 0.08

                        # Y轴反转
                        if self.mouse.config.invert_y:
                            dy = -dy

                        # 死区
                        dist = (dx**2 + dy**2) ** 0.5
                        if dist >= self.mouse.config.deadzone_radius:
                            # 限制单次最大值（防止鼠标被推到屏幕角落触发failsafe）
                            if dist > max_single_move:
                                scale = max_single_move / dist
                                dx *= scale
                                dy *= scale

                            # ★ 4K主屏适配：OBS副屏1080p → 主屏4K的鼠标坐标比例转换
                            # OBS捕获分辨率(副屏) vs 主显示器物理分辨率(4K)
                            # pydirectinput的moveRel是基于主屏像素的，如果游戏在主屏全屏运行，
                            # 鼠标移动量需要按主屏/副屏比例缩放
                            primary_scale = True  # 设为True启用4K主屏比例修正
                            if primary_scale:
                                # 获取主显示器实际分辨率
                                try:
                                    import ctypes
                                    user32 = ctypes.windll.user32
                                    main_w = user32.GetSystemMetrics(0)  # 主屏宽
                                    main_h = user32.GetSystemMetrics(1)  # 主屏高
                                    scale_x = main_w / self.frame_width
                                    scale_y = main_h / self.frame_height
                                    dx *= scale_x
                                    dy *= scale_y
                                except Exception:
                                    pass  # 获取失败则保持原值

                            self.mouse.move_mouse(dx, dy)
                    else:
                        self.current_target = None
                else:
                    self.current_target = None
                    self.mouse.reset_smoothing()

                # 5. 预览窗口
                if self.config.SHOW_PREVIEW:
                    display_frame = frame.copy()

                    if self.config.DRAW_DETECTIONS:
                        # 只显示当前锁定的目标框（高置信度红色框）
                        if self.current_target is not None:
                            display_frame = self.detector.draw_detections(
                                display_frame,
                                [self.current_target],  # 只传当前锁定目标
                                highlight_target=self.current_target,
                                frame_center=frame_center
                            )

                    # 绘制左下角黑色半透明信息背板，方便查看
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (5, self.frame_height - 110), (550, self.frame_height - 5), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

                    # 第一行：总开关状态
                    switch_str = "[1] System Auto-Aim: ENABLED" if activated else "[1] System Auto-Aim: DISABLED"
                    switch_color = (0, 255, 0) if activated else (0, 0, 255)
                    cv2.putText(display_frame, switch_str, (15, self.frame_height - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, switch_color, 2)
                    
                    # 第二行：人物检测及可信度
                    if self.current_target:
                        # 有锁定目标
                        target_str = f"[2] Target: LOCKED | Confidence: {self.current_target.confidence:.2f}"
                        target_color = (0, 255, 255)  # 黄色
                    else:
                        num_det = len(detections) if detections else 0
                        if num_det > 0:
                            # 看到人，但可能在死区或没满足条件
                            target_str = f"[2] Target: {num_det} Person(s) Detected (Validating...)"
                            target_color = (150, 255, 150)
                        else:
                            # 没看到人
                            target_str = "[2] Target: NONE"
                            target_color = (150, 150, 150)
                    cv2.putText(display_frame, target_str, (15, self.frame_height - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)

                    # 第三行：鼠标信号输出状态
                    mouse_output = (activated and self.current_target is not None)
                    if mouse_output:
                        lx, ly = self.mouse.last_offset
                        mouse_str = f"[3] Mouse Signal: YES | dx={lx:.0f} dy={ly:.0f}"
                        mouse_color = (0, 255, 0)  # 绿色
                    else:
                        mouse_str = "[3] Mouse Signal: NO (Idle)"
                        mouse_color = (150, 150, 150)  # 灰色
                    cv2.putText(display_frame, mouse_str, (15, self.frame_height - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouse_color, 2)

                    # 全屏预览：拉满整个屏幕（F9退出）
                    # 获取主显示器分辨率，把OBS画面拉伸铺满
                    try:
                        import ctypes
                        user32 = ctypes.windll.user32
                        screen_w = user32.GetSystemMetrics(0)
                        screen_h = user32.GetSystemMetrics(1)
                    except Exception:
                        screen_w, screen_h = 1920, 1080

                    display_frame = cv2.resize(display_frame, (screen_w, screen_h))
                    cv2.namedWindow("FPS Aim Assist - F8瞄准 F9退出", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("FPS Aim Assist - F8瞄准 F9退出", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("FPS Aim Assist - F8瞄准 F9退出", display_frame)

                # 6. 处理键盘输入（全部用keyboard库，避免cv2.waitKey冲突）
                key = cv2.waitKey(1) & 0xFF
                # cv2.waitKey只用来让imshow刷新画面，退出逻辑全部走keyboard
                if key == 27:
                    print("\n[退出] 原因: ESC键 (cv2)")
                    break
                elif key == ord('q'):
                    print("\n[退出] 原因: Q键 (cv2)")
                    break
                elif key == ord('1'):
                    # 切换策略到最近敌人
                    self.mouse.config.strategy = AimStrategy.NEAREST
                    self.config.AIM_STRATEGY = "nearest"
                    print("[策略] 切换到: 最近敌人")
                elif key == ord('2'):
                    # 切换策略到最大敌人
                    self.mouse.config.strategy = AimStrategy.LARGEST
                    self.config.AIM_STRATEGY = "largest"
                    print("[策略] 切换到: 最大敌人")
                elif key == ord('3'):
                    # 切换策略到最高置信度
                    self.mouse.config.strategy = AimStrategy.HIGHEST_CONFIDENCE
                    self.config.AIM_STRATEGY = "highest"
                    print("[策略] 切换到: 最高置信度")

                # 7. 计算FPS
                current_time = time.time()
                elapsed = current_time - self.fps_timer
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_timer = current_time

                    # 打印状态
                    avg_inf_time = self.detector.get_average_inference_time()
                    aim_status = "ON " if activated else "OFF"
                    print(f"\r[运行] FPS: {self.fps:.1f} | 推理: {avg_inf_time:.1f}ms | 瞄准: {aim_status} | 检测: {len(detections) if 'detections' in dir() else 0}个",
                          end="", flush=True)

                # 控制帧率
                elapsed_loop = time.time() - loop_start
                sleep_time = (1.0 / self.config.TARGET_FPS) - elapsed_loop
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            debug_keys = self._get_debug_key_info()
            print(f"\n[错误] 主循环异常: {e} | 最后按键: [{debug_keys}]")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """停止系统"""
        self.is_running = False
        print("\n[清理] 正在关闭...")
        self.capture.stop()
        cv2.destroyAllWindows()
        print("[清理] 已完全关闭")


def main():
    """入口函数"""
    assist = AimAssist()
    assist.run()


if __name__ == "__main__":
    main()
