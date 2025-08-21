# -*- coding: utf-8 -*-

"""
主GUI应用程序模块

该文件使用Tkinter创建主应用程序窗口，
并管理后台的捕捉线程。
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import queue
import cv2
import os
import numpy as np

from capture_thread import CaptureThread
from config import CHARACTER_PROFILES, SAMPLE_KEYPOINTS
from drawing import draw_character

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("动画桌面捕捉 v2.0")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.frame_queue = queue.Queue()
        self.settings_queue = queue.Queue()

        self.capture_thread = CaptureThread(self.frame_queue, self.settings_queue)
        self.capture_thread.start()

        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(main_frame)
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.create_settings_panel(main_frame)

        self.update_frame()
        self.update_preview()

    def create_settings_panel(self, parent):
        settings_frame = ttk.Frame(parent, padding="10")
        settings_frame.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(settings_frame, text="角色外观 (Profile):").pack(pady=(0, 5), anchor="w")
        self.profile_var = tk.StringVar()
        profile_combobox = ttk.Combobox(settings_frame, textvariable=self.profile_var, state="readonly")
        profile_combobox['values'] = list(CHARACTER_PROFILES.keys())
        profile_combobox.current(0)
        profile_combobox.pack(fill=tk.X)
        profile_combobox.bind("<<ComboboxSelected>>", self.on_profile_select)

        ttk.Label(settings_frame, text="背景 (Background):").pack(pady=(20, 5), anchor="w")
        self.background_var = tk.StringVar()
        background_combobox = ttk.Combobox(settings_frame, textvariable=self.background_var, state="readonly")

        backgrounds = ["black", "blue"]
        if os.path.exists("background.jpg"): backgrounds.append("custom")

        background_combobox['values'] = backgrounds
        background_combobox.current(0)
        background_combobox.pack(fill=tk.X)
        background_combobox.bind("<<ComboboxSelected>>", self.on_background_select)

        ttk.Label(settings_frame, text="置信度阈值 (Confidence):").pack(pady=(20, 5), anchor="w")
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_label = ttk.Label(settings_frame, text=f"{self.confidence_var.get():.2f}")
        self.confidence_label.pack()
        confidence_slider = ttk.Scale(settings_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.confidence_var, command=self.on_threshold_change)
        confidence_slider.pack(fill=tk.X)

        # --- 平滑度控制滑块 ---
        ttk.Label(settings_frame, text="动画平滑度 (Smoothness):").pack(pady=(20, 5), anchor="w")
        self.smoothness_var = tk.DoubleVar(value=0.7) # 对应 alpha=0.3
        self.smoothness_label = ttk.Label(settings_frame, text=f"{self.smoothness_var.get():.2f}")
        self.smoothness_label.pack()
        smoothness_slider = ttk.Scale(settings_frame, from_=0.0, to=0.95, orient=tk.HORIZONTAL, variable=self.smoothness_var, command=self.on_smoothness_change)
        smoothness_slider.pack(fill=tk.X)


        ttk.Label(settings_frame, text="外观预览:").pack(pady=(20, 5), anchor="w")
        self.preview_label = ttk.Label(settings_frame, background="black")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def update_preview(self, profile_name=None):
        if profile_name is None: profile_name = self.profile_var.get()
        profile = CHARACTER_PROFILES[profile_name]
        preview_w, preview_h = 150, 200
        preview_canvas = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
        scaled_kpts = SAMPLE_KEYPOINTS.copy()
        scaled_kpts[:, 0] *= preview_w
        scaled_kpts[:, 1] *= preview_h
        draw_character(preview_canvas, scaled_kpts, profile)
        img = Image.fromarray(cv2.cvtColor(preview_canvas, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.preview_label.imgtk = imgtk
        self.preview_label.configure(image=imgtk)

    def on_profile_select(self, event=None):
        profile_name = self.profile_var.get()
        if profile_name:
            self.settings_queue.put({"profile": profile_name})
            self.update_preview(profile_name)

    def on_background_select(self, event=None):
        bg_name = self.background_var.get()
        if bg_name: self.settings_queue.put({"background": bg_name})

    def on_threshold_change(self, value):
        threshold = self.confidence_var.get()
        self.confidence_label.config(text=f"{threshold:.2f}")
        self.settings_queue.put({"confidence_threshold": threshold})

    def on_smoothness_change(self, value):
        smoothness = self.smoothness_var.get()
        self.smoothness_label.config(text=f"{smoothness:.2f}")
        # 平滑度越高，alpha越小，姿态变化越慢
        alpha = 1.0 - smoothness
        self.settings_queue.put({"smoothing_alpha": alpha})

    def update_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            label_width, label_height = self.video_label.winfo_width(), self.video_label.winfo_height()
            if label_width > 1 and label_height > 1: img.thumbnail((label_width, label_height), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except queue.Empty:
            pass
        finally:
            self.root.after(15, self.update_frame)

    def on_closing(self):
        print("主GUI：正在关闭应用程序...")
        self.capture_thread.stop()
        self.capture_thread.join(timeout=2)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
