# -*- coding: utf-8 -*-

"""
捕捉与处理线程模块

该文件定义了在后台运行的核心工作线程。
它负责屏幕捕捉、姿态估计和图像绘制，
并将最终的图像帧放入队列，以供GUI线程消费。
"""

import threading
import mss
import numpy as np
import cv2
from ultralytics import YOLO
from drawing import draw_character, draw_hud
from config import CHARACTER_PROFILES

class CaptureThread(threading.Thread):
    def __init__(self, frame_queue, settings_queue):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.settings_queue = settings_queue
        self._stop_event = threading.Event()
        self.model = YOLO("yolo11n-pose.pt")
        print("后台线程：YOLOv8模型已加载。")

    def stop(self):
        """发送停止信号给线程。"""
        self._stop_event.set()

    def run(self):
        """线程的主执行循环。"""
        # --- 初始化 ---
        is_recording = False # 录制功能暂未在GUI中实现
        video_writer = None

        # 初始化默认设置
        profile_names = list(CHARACTER_PROFILES.keys())
        current_profile_name = profile_names[0]
        backgrounds = ["black", "blue"]
        current_bg_name = backgrounds[0]
        current_confidence_threshold = 0.5

        # 尝试加载自定义背景
        if cv2.os.path.exists("background.jpg"):
            custom_bg = cv2.imread("background.jpg")
            backgrounds.append("custom")
        else:
            custom_bg = None

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            monitor_width, monitor_height = monitor["width"], monitor["height"]

            if custom_bg is not None:
                custom_bg = cv2.resize(custom_bg, (monitor_width, monitor_height))

            # --- 主循环 ---
            while not self._stop_event.is_set():
                # --- 检查来自GUI的设置更新 ---
                if not self.settings_queue.empty():
                    settings = self.settings_queue.get()
                    current_profile_name = settings.get("profile", current_profile_name)
                    current_bg_name = settings.get("background", current_bg_name)
                    current_confidence_threshold = settings.get("confidence_threshold", current_confidence_threshold)

                # --- 设置当前帧的画布 ---
                current_profile = CHARACTER_PROFILES[current_profile_name]
                if current_bg_name == "black":
                    animation_canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)
                elif current_bg_name == "blue":
                    animation_canvas = np.full((monitor_height, monitor_width, 3), (139, 0, 0), dtype=np.uint8)
                elif current_bg_name == "custom" and custom_bg is not None:
                    animation_canvas = custom_bg.copy()
                else: # Fallback
                    animation_canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)

                # --- 核心处理逻辑 ---
                sct_img = sct.grab(monitor)
                img_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                results = self.model(img_bgr, stream=True, verbose=False, conf=current_confidence_threshold)

                for r in results:
                    for person_keypoints in r.keypoints.data.cpu().numpy():
                        draw_character(animation_canvas, person_keypoints, current_profile, confidence_threshold=current_confidence_threshold)

                # --- 绘制UI ---
                draw_hud(animation_canvas, is_recording, current_profile_name, current_bg_name)

                # --- 将结果放入队列 ---
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(animation_canvas)

        print("后台线程：已停止。")
