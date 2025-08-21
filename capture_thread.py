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
        self.last_known_poses = {} # 用于存储每个跟踪ID的最后姿态
        print("后台线程：YOLOv8模型已加载。")

    def stop(self):
        self._stop_event.set()

    def run(self):
        is_recording = False
        video_writer = None

        current_profile_name = list(CHARACTER_PROFILES.keys())[0]
        backgrounds = ["black", "blue"]
        current_bg_name = backgrounds[0]
        current_confidence_threshold = 0.5
        smoothing_alpha = 0.3 # 插值系数 (lerp alpha)

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

            while not self._stop_event.is_set():
                if not self.settings_queue.empty():
                    settings = self.settings_queue.get()
                    current_profile_name = settings.get("profile", current_profile_name)
                    current_bg_name = settings.get("background", current_bg_name)
                    current_confidence_threshold = settings.get("confidence_threshold", current_confidence_threshold)
                    smoothing_alpha = settings.get("smoothing_alpha", smoothing_alpha)

                current_profile = CHARACTER_PROFILES[current_profile_name]
                if current_bg_name == "black": canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)
                elif current_bg_name == "blue": canvas = np.full((monitor_height, monitor_width, 3), (139, 0, 0), dtype=np.uint8)
                elif current_bg_name == "custom" and custom_bg is not None: canvas = custom_bg.copy()
                else: canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)

                sct_img = sct.grab(monitor)
                img_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

                results = self.model.track(img_bgr, persist=True, verbose=False, conf=current_confidence_threshold)

                track_ids = results[0].boxes.id
                all_keypoints = results[0].keypoints.data.cpu().numpy()

                current_frame_ids = set()

                if track_ids is not None:
                    for i, person_keypoints in enumerate(all_keypoints):
                        track_id = int(track_ids[i])
                        current_frame_ids.add(track_id)

                        if track_id in self.last_known_poses:
                            smoothed_kpts = self.last_known_poses[track_id] * (1 - smoothing_alpha) + person_keypoints * smoothing_alpha
                        else:
                            smoothed_kpts = person_keypoints

                        self.last_known_poses[track_id] = smoothed_kpts

                        draw_character(canvas, smoothed_kpts, current_profile, confidence_threshold=current_confidence_threshold)

                stale_ids = set(self.last_known_poses.keys()) - current_frame_ids
                for stale_id in stale_ids:
                    del self.last_known_poses[stale_id]

                draw_hud(canvas, is_recording, current_profile_name, current_bg_name)

                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(canvas)

        print("后台线程：已停止。")
