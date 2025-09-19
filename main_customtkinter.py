# -*- coding: utf-8 -*-

"""
主GUI应用程序模块 (CustomTkinter版)

该文件使用CustomTkinter创建主应用程序窗口，
并管理后台的捕捉线程。
"""

import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import queue
import cv2
import os
import numpy as np

from capture_thread import CaptureThread
from config import CHARACTER_PROFILES, SAMPLE_KEYPOINTS
from drawing import draw_character
from roi_selector import ROISelector

# 设置App主题
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("动画桌面捕捉 v3.0")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.geometry("1200x720")

        self.capture_region = None
        self.selected_model = "yolo11n-pose.pt"
        self.is_recording = False
        self.frame_queue = queue.Queue()
        self.settings_queue = queue.Queue()
        self.capture_thread = None

        self.start_capture_thread()

        # --- GUI 布局 ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=0) # Settings panel should not expand
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.main_frame, text="正在启动...")
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.create_settings_panel(self.main_frame)

        self.update_frame()
        self.update_preview()
        self.on_filter_param_change(None)

    def start_capture_thread(self):
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.stop()
            self.capture_thread.join(timeout=1)
        self.capture_thread = CaptureThread(
            frame_queue=self.frame_queue,
            settings_queue=self.settings_queue,
            capture_region=self.capture_region,
            model_name=self.selected_model
        )
        self.capture_thread.start()
        print(f"主GUI：捕捉线程已启动 (模型: {self.selected_model})。")

    def create_settings_panel(self, parent):
        settings_frame = ctk.CTkScrollableFrame(parent, label_text="设置面板")
        settings_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ns")
        settings_frame.grid_columnconfigure(0, weight=1)

        # --- 捕捉与模型设置 ---
        capture_frame = ctk.CTkFrame(settings_frame)
        capture_frame.pack(fill=tk.X, pady=(5, 10))
        capture_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(capture_frame, text="捕捉与模型", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=5, sticky="w")

        ctk.CTkButton(capture_frame, text="选择捕捉区域", command=self.open_roi_selector).grid(row=1, column=0, sticky="ew", padx=5)
        ctk.CTkButton(capture_frame, text="重置为全屏", command=self.reset_roi).grid(row=2, column=0, sticky="ew", padx=5, pady=(5,0))

        self.model_var = ctk.StringVar(value=self.selected_model)
        ctk.CTkComboBox(capture_frame, variable=self.model_var, state="readonly",
                        values=['yolo11n-pose.pt', 'yolo11s-pose.pt', 'yolo11m-pose.pt'],
                        command=self.on_model_select).grid(row=3, column=0, sticky="ew", padx=5, pady=10)

        # --- 外观与视图设置 ---
        appearance_frame = ctk.CTkFrame(settings_frame)
        appearance_frame.pack(fill=tk.X, pady=10)
        appearance_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(appearance_frame, text="外观与视图", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=5, sticky="w")

        self.view_mode_var = ctk.StringVar(value="Animation View")
        ctk.CTkRadioButton(appearance_frame, text="动画视图", variable=self.view_mode_var, value="Animation View", command=self.on_view_mode_change).grid(row=1, column=0, sticky="w", padx=10)
        ctk.CTkRadioButton(appearance_frame, text="视频骨骼预览", variable=self.view_mode_var, value="Debug View", command=self.on_view_mode_change).grid(row=2, column=0, sticky="w", padx=10)

        self.profile_var = ctk.StringVar()
        ctk.CTkComboBox(appearance_frame, variable=self.profile_var, state="readonly",
                        values=list(CHARACTER_PROFILES.keys()), command=self.on_profile_select).grid(row=3, column=0, sticky="ew", padx=5, pady=10)

        # --- 效果与参数调整 ---
        params_frame = ctk.CTkFrame(settings_frame)
        params_frame.pack(fill=tk.X, pady=10)
        params_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(params_frame, text="效果与参数", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, pady=5, sticky="w")

        ctk.CTkLabel(params_frame, text="置信度:").grid(row=1, column=0, sticky="w", padx=5)
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_label = ctk.CTkLabel(params_frame, text=f"{self.confidence_var.get():.2f}", width=40)
        self.confidence_label.grid(row=1, column=2)
        ctk.CTkSlider(params_frame, from_=0.0, to=1.0, variable=self.confidence_var, command=self.on_threshold_change).grid(row=1, column=1, sticky="ew")

        ctk.CTkLabel(params_frame, text="Min Cutoff:").grid(row=2, column=0, sticky="w", padx=5)
        self.min_cutoff_var = tk.DoubleVar(value=1.0)
        self.min_cutoff_label = ctk.CTkLabel(params_frame, text=f"{self.min_cutoff_var.get():.2f}", width=40)
        self.min_cutoff_label.grid(row=2, column=2)
        ctk.CTkSlider(params_frame, from_=0.1, to=2.0, variable=self.min_cutoff_var, command=self.on_filter_param_change).grid(row=2, column=1, sticky="ew")

        ctk.CTkLabel(params_frame, text="Beta:").grid(row=3, column=0, sticky="w", padx=5)
        self.beta_var = tk.DoubleVar(value=0.7)
        self.beta_label = ctk.CTkLabel(params_frame, text=f"{self.beta_var.get():.2f}", width=40)
        self.beta_label.grid(row=3, column=2)
        ctk.CTkSlider(params_frame, from_=0.0, to=1.5, variable=self.beta_var, command=self.on_filter_param_change).grid(row=3, column=1, sticky="ew")

        # --- 预览和录制 ---
        preview_frame = ctk.CTkFrame(settings_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        ctk.CTkLabel(preview_frame, text="外观预览").pack()
        self.preview_label = ctk.CTkLabel(preview_frame, text="")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        self.record_button = ctk.CTkButton(settings_frame, text="开始录制", command=self.toggle_recording)
        self.record_button.pack(pady=10, fill=tk.X)

    def open_roi_selector(self):
        self.withdraw()
        self.roi_selector = ROISelector(self, self.roi_selection_callback)

    def reset_roi(self):
        self.roi_selection_callback(None)

    def roi_selection_callback(self, roi):
        self.capture_region = roi
        self.deiconify()
        self.start_capture_thread()

    def on_model_select(self, model_name):
        self.selected_model = model_name
        self.start_capture_thread()

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        self.settings_queue.put({"is_recording": self.is_recording})
        if self.is_recording: self.record_button.configure(text="停止录制", fg_color="red")
        else: self.record_button.configure(text="开始录制", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])

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
        imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(preview_w, preview_h))
        self.preview_label.configure(image=imgtk)

    def on_view_mode_change(self):
        mode = self.view_mode_var.get()
        self.settings_queue.put({"view_mode": mode})

    def on_profile_select(self, profile_name):
        self.settings_queue.put({"profile": profile_name})
        self.update_preview(profile_name)

    def on_background_select(self, bg_name):
        self.settings_queue.put({"background": bg_name})

    def on_threshold_change(self, value):
        self.confidence_label.configure(text=f"{value:.2f}")
        self.settings_queue.put({"confidence_threshold": value})

    def on_filter_param_change(self, value):
        min_cutoff = self.min_cutoff_var.get()
        beta = self.beta_var.get()
        self.min_cutoff_label.configure(text=f"{min_cutoff:.2f}")
        self.beta_label.configure(text=f"{beta:.2f}")
        self.settings_queue.put({"min_cutoff": min_cutoff, "beta": beta})

    def update_frame(self):
        try:
            frame = self.frame_queue.get_nowait()
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(frame.shape[1], frame.shape[0]))
            self.video_label.configure(image=imgtk, text="")
        except queue.Empty:
            pass
        finally:
            self.after(15, self.update_frame)

    def on_closing(self):
        print("主GUI：正在关闭应用程序...")
        self.capture_thread.stop()
        self.capture_thread.join(timeout=2)
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()
