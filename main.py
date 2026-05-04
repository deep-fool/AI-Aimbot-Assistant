##imgiz=320
import cv2
import numpy as np
import torch
import time
import serial
import keyboard
import threading
from dataclasses import dataclass, field
from queue import Queue, Empty
from pathlib import Path
from pynput import mouse
from ultralytics import YOLO
import mss
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES  # type: ignore[import-not-found]

    HAS_DND = True
except Exception:
    TkinterDnD = tk.Tk
    DND_FILES = None
    HAS_DND = False


def round_up_to_multiple(value: int, multiple: int = 32) -> int:
    value = max(1, int(value))
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def get_default_inference_size(model, fallback: int = 320) -> int:
    """Return a model-friendly inference size that does not scale with capture radius."""
    try:
        overrides = getattr(model, "overrides", {}) or {}
        raw_size = overrides.get("imgsz", fallback)
        if isinstance(raw_size, (list, tuple)):
            raw_size = raw_size[0]
        size = int(raw_size)
    except Exception:
        size = fallback
    return round_up_to_multiple(max(32, size), 32)


@dataclass
class RuntimeConfig:
    enabled: bool = True
    right_click_enable: bool = True
    auto_trigger: bool = True
    debug: bool = False
    capture_radius: int = 320
    conf_thres: float = 0.4
    lock_dist: int = 70
    fire_dist: int = 7
    recoil_y: int = 0
    send_interval: float = 0.005
    serial_port: str = "COM5"
    baud_rate: int = 115200
    model_path: str = ""
    class_ids: str = "0"
    use_gpu: bool = True
    a: int = 3


class ConfigStore:
    def __init__(self):
        self._cfg = RuntimeConfig()
        self._lock = threading.RLock()

    def snapshot(self) -> RuntimeConfig:
        with self._lock:
            return RuntimeConfig(**self._cfg.__dict__)

    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._cfg, key):
                    setattr(self._cfg, key, value)

    def set_model_path(self, model_path: str):
        self.update(model_path=model_path)


class ModelManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._model = None
        self._path = ""
        self._error = ""

    def ensure_loaded(self, model_path: str):
        model_path = (model_path or "").strip()
        if not model_path:
            with self._lock:
                self._model = None
                self._path = ""
                self._error = "请拖入或选择模型文件"
            return None, self._error

        with self._lock:
            if model_path == self._path and self._model is not None:
                return self._model, self._error

        if not Path(model_path).exists():
            with self._lock:
                self._model = None
                self._path = ""
                self._error = f"模型不存�?: {model_path}"
            return None, self._error

        try:
            new_model = YOLO(model_path, task="detect")
            with self._lock:
                self._model = new_model
                self._path = model_path
                self._error = ""
            return new_model, ""
        except Exception as exc:
            with self._lock:
                self._model = None
                self._path = ""
                self._error = f"妯″瀷鍔犺浇澶辫�?: {exc}"
            return None, self._error


class AimAssistApp:
    def __init__(self):
        self.exit_event = threading.Event()
        self.config = ConfigStore()
        self.model_manager = ModelManager()
        self.serial_queue = Queue(maxsize=2)
        self.latest_frame = None
        self.right_pressed = False
        self.last_target_abs = None
        self.last_missing_frames = 0
        self.last_send_time = 0.0
        self.screen_width = 0
        self.screen_height = 0
        self.cx_scr = 0
        self.cy_scr = 0
        self.monitor_lock = threading.RLock()
        self.ui_state_lock = threading.RLock()
        self.grab_monitor = {"left": 0, "top": 0, "width": 1, "height": 1}
        self.pending_model_status = None
        self.status_text = None
        self.current_fps = 0.0
        self.fps_lock = threading.RLock()
        self._build_ui()
        self._init_screen_metrics()
        self._init_serial()
        self._start_workers()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(120, self._refresh_status)

    def _build_ui(self):
        self.root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
        self.root.title("Aim Assist 控制面板 v1.0")
        self.root.geometry("700x850")
        self.root.minsize(700, 800)
        self.root.attributes("-topmost", True)
        self.status_text = tk.StringVar(master=self.root, value="就绪")
        self.fps_var = tk.StringVar(master=self.root, value="FPS: 0")

        # 设置现代配色方案
        bg_color = "#1e1e2e"
        fg_color = "#e0e0e0"
        accent_color = "#0d7377"

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color)
        style.configure('TLabelframe', background=bg_color, foreground=fg_color)
        style.configure('TLabelframe.Label', background=bg_color, foreground=accent_color)
        style.configure('TCheckbutton', background=bg_color, foreground=fg_color)
        style.configure('TButton', background=accent_color)
        style.map('TButton', background=[('active', '#14a085')])

        self.root.configure(bg=bg_color)

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        # 顶部标题和状态栏
        header = ttk.Frame(main)
        header.pack(fill="x", pady=(0, 12))

        title = ttk.Label(header, text="🎯 Aim Assist 控制面板", font=("Microsoft YaHei UI", 16, "bold"))
        title.pack(anchor="w", side="left")

        # FPS 显示
        self.fps_var = tk.StringVar(value="FPS: 0")
        fps_label = ttk.Label(header, textvariable=self.fps_var, font=("Microsoft YaHei UI", 12, "bold"),
                              foreground="#14a085")
        fps_label.pack(anchor="e", side="right", padx=10)

        # 分隔�?
        sep1 = ttk.Label(main, text="─" * 60, foreground="#444444")
        sep1.pack(anchor="w", pady=(0, 8))

        cfg = self.config.snapshot()

        toggles = ttk.LabelFrame(main, text="开�?")
        toggles.pack(fill="x", pady=6)

        self.enabled_var = tk.BooleanVar(value=cfg.enabled)
        self.right_click_enable_var = tk.BooleanVar(value=cfg.right_click_enable)
        self.auto_trigger_var = tk.BooleanVar(value=cfg.auto_trigger)
        self.debug_var = tk.BooleanVar(value=cfg.debug)

        for row, (label, var, key) in enumerate([
            ("启用", self.enabled_var, "enabled"),
            ("右键开�?/按住生效", self.right_click_enable_var, "right_click_enable"),
            ("自动扳机", self.auto_trigger_var, "auto_trigger"),
            ("调试显示", self.debug_var, "debug"),
        ]):
            cb = ttk.Checkbutton(toggles, text=label, variable=var, command=lambda k=key, v=var: self._sync_bool(k, v))
            cb.grid(row=row // 2, column=row % 2, sticky="w", padx=10, pady=6)

        params = ttk.LabelFrame(main, text="参数")
        params.pack(fill="both", expand=False, pady=6)

        self.capture_radius_var = tk.IntVar(value=cfg.capture_radius)
        self.conf_var = tk.DoubleVar(value=cfg.conf_thres)
        self.lock_dist_var = tk.IntVar(value=cfg.lock_dist)
        self.fire_dist_var = tk.IntVar(value=cfg.fire_dist)
        self.recoil_var = tk.IntVar(value=cfg.recoil_y)
        self.a_var = tk.IntVar(value=cfg.a)
        self.send_interval_var = tk.DoubleVar(value=cfg.send_interval)
        self.serial_port_var = tk.StringVar(value=cfg.serial_port)
        self.baud_rate_var = tk.IntVar(value=cfg.baud_rate)
        self.class_ids_var = tk.StringVar(value=cfg.class_ids)
        self.model_path_var = tk.StringVar(value=cfg.model_path)

        self._add_slider(params, "识别范围半径", self.capture_radius_var, 64, 900, 1, "capture_radius")
        self._add_slider(params, "置信度阈�?", self.conf_var, 0.05, 0.95, 0.01, "conf_thres")
        self._add_slider(params, "锁定距离", self.lock_dist_var, 5, 300, 1, "lock_dist")
        self._add_slider(params, "自动扳机距离", self.fire_dist_var, 0, 80, 1, "fire_dist")
        self._add_slider(params, "压枪补偿Y", self.recoil_var, -50, 50, 1, "recoil_y")
        self._add_slider(params, "目标丢失阈�? a", self.a_var, 1, 100, 1, "a")
        self._add_slider(params, "发送间�?(ms)", self.send_interval_var, 1, 50, 1, "send_interval_ms", scale_value=True)

        serial_frame = ttk.LabelFrame(main, text="连接与识�?")
        serial_frame.pack(fill="x", pady=6)

        self._add_entry(serial_frame, "串口", self.serial_port_var, 0)
        self._add_entry(serial_frame, "波特�?", self.baud_rate_var, 1)
        self._add_entry(serial_frame, "目标类别ID", self.class_ids_var, 2)

        path_frame = ttk.LabelFrame(main, text="模型路径")
        path_frame.pack(fill="both", expand=True, pady=6)

        path_row = ttk.Frame(path_frame)
        path_row.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Entry(path_row, textvariable=self.model_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(path_row, text="浏览", command=self._browse_model).pack(side="left", padx=(8, 0))

        self.drop_hint = ttk.Label(
            path_frame,
            text="把模型文件拖到下面这个区域，或点浏览选择\n支持 .engine / .pt / .onnx �? Ultralytics 支持的格�?",
            anchor="center",
            justify="center",
            padding=10,
        )
        self.drop_hint.pack(fill="both", expand=True, padx=8, pady=8)
        if HAS_DND:
            self.drop_hint.drop_target_register(DND_FILES)
            self.drop_hint.dnd_bind("<<Drop>>", self._on_drop_model)

        self.model_status = ttk.Label(path_frame, text="未加载模�?", foreground="#666")
        self.model_status.pack(fill="x", padx=8, pady=(0, 8))

        # 底部状态栏
        sep2 = ttk.Label(main, text="─" * 60, foreground="#444444")
        sep2.pack(anchor="w", pady=(8, 6))

        bottom = ttk.Frame(main)
        bottom.pack(fill="x", side="bottom", pady=(6, 0))
        status_label = ttk.Label(bottom, textvariable=self.status_text, font=("Microsoft YaHei UI", 10))
        status_label.pack(side="left", expand=True, fill="x")
        ttk.Button(bottom, text="�? 退�?", command=self.on_close, width=8).pack(side="right", padx=(4, 0))

        self.model_path_var.trace_add("write", lambda *_: self._sync_string("model_path", self.model_path_var))
        self.serial_port_var.trace_add("write", lambda *_: self._sync_string("serial_port", self.serial_port_var))
        self.class_ids_var.trace_add("write", lambda *_: self._sync_string("class_ids", self.class_ids_var))
        self.baud_rate_var.trace_add("write", lambda *_: self._sync_int("baud_rate", self.baud_rate_var))
        self.capture_radius_var.trace_add("write", lambda *_: self._sync_int("capture_radius", self.capture_radius_var))
        self.lock_dist_var.trace_add("write", lambda *_: self._sync_int("lock_dist", self.lock_dist_var))
        self.fire_dist_var.trace_add("write", lambda *_: self._sync_int("fire_dist", self.fire_dist_var))
        self.recoil_var.trace_add("write", lambda *_: self._sync_int("recoil_y", self.recoil_var))
        self.a_var.trace_add("write", lambda *_: self._sync_int("a", self.a_var))
        self.conf_var.trace_add("write", lambda *_: self._sync_float("conf_thres", self.conf_var))
        self.send_interval_var.trace_add("write", lambda *_: self._sync_float("send_interval",
                                                                              self.send_interval_var.get() / 1000.0,
                                                                              direct=True))

    def _add_slider(self, parent, label, var, frm, to, resolution, key, scale_value=False):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=label, width=16).pack(side="left")
        if scale_value:
            scale = tk.Scale(row, from_=frm, to=to, orient="horizontal", resolution=resolution, variable=var,
                             showvalue=True, length=300)
        else:
            scale = tk.Scale(row, from_=frm, to=to, orient="horizontal", resolution=resolution, variable=var,
                             showvalue=True, length=300)
        scale.pack(side="left", fill="x", expand=True)
        value_label = ttk.Label(row, width=12)
        value_label.pack(side="left", padx=6)

        def refresh_label(*_):
            value = var.get()
            if key == "send_interval_ms":
                value_label.config(text=f"{int(value)} ms")
            elif isinstance(value, float):
                value_label.config(text=f"{value:.2f}")
            else:
                value_label.config(text=str(value))
            if key == "send_interval_ms":
                self.config.update(send_interval=max(0.001, float(value) / 1000.0))
            elif key == "capture_radius":
                self.config.update(capture_radius=max(32, int(value)))
            elif key == "conf_thres":
                self.config.update(conf_thres=float(value))
            elif key == "lock_dist":
                self.config.update(lock_dist=int(value))
            elif key == "fire_dist":
                self.config.update(fire_dist=int(value))
            elif key == "recoil_y":
                self.config.update(recoil_y=int(value))
            elif key == "a":
                try:
                    self.config.update(a=max(1, int(value)))
                except Exception:
                    pass

        var.trace_add("write", refresh_label)
        refresh_label()

    def _add_entry(self, parent, label, var, row_index):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=label, width=16).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

    def _sync_bool(self, key, var):
        self.config.update(**{key: bool(var.get())})

    def _sync_string(self, key, var):
        self.config.update(**{key: str(var.get())})

    def _sync_int(self, key, var):
        try:
            self.config.update(**{key: int(var.get())})
        except Exception:
            pass

    def _sync_float(self, key, value=None, direct=False):
        try:
            if direct:
                self.config.update(**{key: float(value)})
            else:
                self.config.update(**{key: float(value.get())})
        except Exception:
            pass

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[
                ("模型文件", "*.engine *.pt *.onnx *.xml *.bin"),
                ("全部文件", "*.*"),
            ],
        )
        if path:
            self.model_path_var.set(path)

    def _on_drop_model(self, event):
        try:
            paths = self.root.tk.splitlist(event.data)
            if paths:
                self.model_path_var.set(paths[0])
        except Exception:
            raw = str(event.data).strip().strip("{}")
            if raw:
                self.model_path_var.set(raw)

    def _init_screen_metrics(self):
        with mss.mss() as temp_sct:
            monitor_info = temp_sct.monitors[1]
            self.screen_width, self.screen_height = monitor_info["width"], monitor_info["height"]
        self.cx_scr, self.cy_scr = self.screen_width // 2, self.screen_height // 2
        self._update_grab_monitor(self.config.snapshot().capture_radius)

    def _update_grab_monitor(self, capture_radius: int):
        left = max(0, self.cx_scr - capture_radius)
        top = max(0, self.cy_scr - capture_radius)
        right = min(self.screen_width, self.cx_scr + capture_radius)
        bottom = min(self.screen_height, self.cy_scr + capture_radius)
        with self.monitor_lock:
            self.grab_monitor = {"left": left, "top": top, "width": max(1, right - left),
                                 "height": max(1, bottom - top)}

    def _init_serial(self):
        self.ser = None
        self._reopen_serial(self.config.snapshot())

    def _reopen_serial(self, cfg: RuntimeConfig):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        try:
            self.ser = serial.Serial(cfg.serial_port, cfg.baud_rate, timeout=0, write_timeout=0)
            time.sleep(1.2)
            self.status_text.set(f"串口已连�?: {cfg.serial_port}")
        except Exception as exc:
            self.ser = None
            self.status_text.set(f"串口打开失败: {exc}")

    def _start_workers(self):
        threading.Thread(target=self._serial_worker, daemon=True).start()
        threading.Thread(target=self._capture_worker, daemon=True).start()
        threading.Thread(target=self._inference_worker, daemon=True).start()
        threading.Thread(target=self._mouse_worker, daemon=True).start()

    def _serial_worker(self):
        while not self.exit_event.is_set():
            try:
                data = self.serial_queue.get(timeout=0.1)
                x, y, fire = data
                if self.ser and self.ser.is_open:
                    msg = f"{int(x)},{int(y)},{int(fire)}\n".encode("ascii")
                    self.ser.write(msg)
            except Empty:
                continue
            except Exception:
                continue

    def _send_to_arduino_async(self, x, y, fire, cfg: RuntimeConfig):
        now = time.time()
        if now - self.last_send_time < max(0.001, cfg.send_interval):
            return
        self.last_send_time = now
        if self.serial_queue.full():
            try:
                self.serial_queue.get_nowait()
            except Exception:
                pass
        self.serial_queue.put((x, y, fire))

    def _capture_worker(self):
        with mss.mss() as sct:
            current_radius = None
            while not self.exit_event.is_set():
                cfg = self.config.snapshot()
                if current_radius != cfg.capture_radius:
                    current_radius = cfg.capture_radius
                    self._update_grab_monitor(current_radius)

                with self.monitor_lock:
                    monitor = self.grab_monitor.copy()
                sct_img = sct.grab(monitor)
                img = np.frombuffer(sct_img.bgra, dtype=np.uint8).reshape((monitor["height"], monitor["width"], 4))
                self.latest_frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _mouse_worker(self):
        def on_click(x, y, button, pressed):
            if button == mouse.Button.right:
                self.right_pressed = pressed

        listener = mouse.Listener(on_click=on_click)
        listener.start()
        while not self.exit_event.is_set():
            time.sleep(0.2)
        try:
            listener.stop()
        except Exception:
            pass

    def _parse_classes(self, class_text: str):
        values = []
        for part in str(class_text).replace("�?", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(int(part))
            except Exception:
                continue
        return values or [0]

    def _update_fps(self, fps: float):
        """从推理线程安全地更新 FPS 显示�?"""
        with self.fps_lock:
            self.current_fps = fps
        try:
            self.fps_var.set(f"FPS: {fps:.1f}")
        except Exception:
            pass

    def _inference_worker(self):
        print(f"PyTorch版本: {torch.__version__}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        fps_counter = 0
        fps_start_time = time.time()
        prev_frame_time = time.time()
        current_model_path = ""
        model = None
        print("AI 核心启动，按 P 键退出程�?...")

        while not self.exit_event.is_set():
            if keyboard.is_pressed("p"):
                self.exit_event.set()
                break

            cfg = self.config.snapshot()

            if cfg.capture_radius > 0:
                self._update_grab_monitor(cfg.capture_radius)

            if cfg.model_path != current_model_path:
                model, err = self.model_manager.ensure_loaded(cfg.model_path)
                current_model_path = cfg.model_path.strip()
                if err:
                    with self.ui_state_lock:
                        self.pending_model_status = (err, "#b00020")
                    time.sleep(0.2)
                    continue
                with self.ui_state_lock:
                    self.pending_model_status = (f"已加�?: {Path(current_model_path).name}", "#1a7f37")

            if model is None or self.latest_frame is None:
                time.sleep(0.01)
                continue

            frame_crop = self.latest_frame
            imgsz = get_default_inference_size(model)
            class_ids = self._parse_classes(cfg.class_ids)

            # 统计 FPS（始终统计，不依�? debug 模式�?
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                self._update_fps(current_fps)
                if cfg.debug:
                    print(f"调试FPS: {current_fps:.1f}")
                fps_counter = 0
                fps_start_time = time.time()

            try:
                results = model(
                    frame_crop,
                    verbose=False,
                    conf=cfg.conf_thres,
                    half=(device == "cuda"),
                    imgsz=imgsz,
                    stream=True,
                    classes=class_ids,
                    device=0 if device == "cuda" and cfg.use_gpu else "cpu",
                )
            except Exception as exc:
                with self.ui_state_lock:
                    self.pending_model_status = (f"推理失败: {exc}", "#b00020")
                time.sleep(0.15)
                continue

            with self.monitor_lock:
                monitor = self.grab_monitor.copy()
            left, top = monitor["left"], monitor["top"]

            current_targets_abs = []
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    coords = boxes.xyxy.cpu().numpy()
                    centers_x = (coords[:, 0] + coords[:, 2]) * 0.5 + left
                    centers_y = (coords[:, 1] + coords[:, 3]) * 0.5 + top
                    current_targets_abs = list(zip(centers_x.astype(int), centers_y.astype(int)))

            final_target = None
            final_target_is_detection = False
            if current_targets_abs:
                targets = np.array(current_targets_abs)
                center = np.array([self.cx_scr, self.cy_scr])
                center_dists = np.sum((targets - center) ** 2, axis=1)
                nearest_idx = int(np.argmin(center_dists))

                # 若存在上次目标，尝试保持锁定，只有在连续丢失超过 cfg.a 帧后才彻底放�?
                if self.last_target_abs is not None:
                    last_pos = np.array(self.last_target_abs)
                    last_dists = np.sum((targets - last_pos) ** 2, axis=1)
                    last_idx = int(np.argmin(last_dists))
                    if last_dists[last_idx] < cfg.lock_dist * cfg.lock_dist:
                        # 仍然检测到上次目标，重置丢失计数，并优先使用上次目标（如果它离中心更近的情况下�?
                        self.last_missing_frames = 0
                        if last_dists[last_idx] < center_dists[nearest_idx] * 1.5:
                            final_target = tuple(targets[last_idx])
                            final_target_is_detection = True
                    else:
                        # 本帧未检测到上次目标，增加丢失计数；若尚未超过阈值，则继续保持上次目标的位置
                        self.last_missing_frames += 1
                        if self.last_missing_frames <= cfg.a:
                            final_target = tuple(self.last_target_abs)
                        else:
                            # 超过阈值，彻底丢失
                            self.last_target_abs = None
                            self.last_missing_frames = 0

                if final_target is None:
                    final_target = tuple(targets[nearest_idx])
                    final_target_is_detection = True
            else:
                # 本帧无检测到任何目标：若有上次目标则按阈值决定是否继续保�?
                if self.last_target_abs is not None:
                    self.last_missing_frames += 1
                    if self.last_missing_frames <= cfg.a:
                        final_target = tuple(self.last_target_abs)
                    else:
                        self.last_target_abs = None
                        self.last_missing_frames = 0

            fire = 0
            if final_target:
                dist_sq = (final_target[0] - self.cx_scr) ** 2 + (final_target[1] - self.cy_scr) ** 2
                if cfg.auto_trigger and dist_sq <= cfg.fire_dist * cfg.fire_dist:
                    fire = 1

            active = cfg.enabled and (self.right_pressed if cfg.right_click_enable else True)
            if active:
                if final_target:
                    y_offset = cfg.recoil_y
                    self._send_to_arduino_async(final_target[0], final_target[1] + y_offset, fire, cfg)
                    # 更新上次目标，如果本帧来自真实检测则重置丢失计数
                    self.last_target_abs = final_target
                    if final_target_is_detection:
                        self.last_missing_frames = 0
                else:
                    self._send_to_arduino_async(self.cx_scr, self.cy_scr, 0, cfg)
                    self.last_target_abs = None

            if cfg.debug:
                curr_time = time.time()
                fps = 1 / (curr_time - prev_frame_time + 1e-6)
                prev_frame_time = curr_time
                debug_img = frame_crop.copy()
                for tx_abs, ty_abs in current_targets_abs:
                    tx_rel, ty_rel = tx_abs - left, ty_abs - top
                    color = (0, 0, 255) if (self.last_target_abs and (tx_abs, ty_abs) == self.last_target_abs) else (0,
                                                                                                                     255,
                                                                                                                     0)
                    cv2.circle(debug_img, (tx_rel, ty_rel), 5, color, -1)
                cv2.putText(debug_img, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_img, f"imgsz: {imgsz}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(debug_img, f"targets: {len(current_targets_abs)}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 0, 0), 2)
                cv2.imshow("Aim Assist", debug_img)
                cv2.waitKey(1)

        self.exit_event.set()

    def _refresh_status(self):
        if self.exit_event.is_set():
            return
        cfg = self.config.snapshot()
        self._update_grab_monitor(cfg.capture_radius)
        with self.ui_state_lock:
            pending = self.pending_model_status
            self.pending_model_status = None
        if pending is not None:
            text, color = pending
            self.model_status.config(text=text, foreground=color)
        self.status_text.set(
            f"范围半径 {cfg.capture_radius} | imgsz 自适应 | 模型: {Path(cfg.model_path).name if cfg.model_path else '未加�?'}"
        )
        self.root.after(120, self._refresh_status)

    def on_close(self):
        self.exit_event.set()
        self.root.after(100, self._cleanup_and_exit)

    def _cleanup_and_exit(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = AimAssistApp()
    app.run()