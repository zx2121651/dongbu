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
import time
from ultralytics import YOLO
from drawing import draw_character, draw_hud
from config import CHARACTER_PROFILES
from one_euro_filter import OneEuroFilter

class CaptureThread(threading.Thread):
    def __init__(self, frame_queue, settings_queue, capture_region=None, model_name="yolo11n-pose.pt"):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.settings_queue = settings_queue
        self.capture_region = capture_region
        self.model_name = model_name
        self._stop_event = threading.Event()
        self.model = YOLO(self.model_name)
        self.pose_filters = {} # 存储每个跟踪ID的滤波器组
        print(f"后台线程：YOLOv8模型 {self.model_name} 已加载。")

    def stop(self):
        self._stop_event.set()

    def run(self):
        is_recording = False
        video_writer = None

        current_profile_name = list(CHARACTER_PROFILES.keys())[0]
        current_bg_name = "black"
        current_confidence_threshold = 0.5
        current_view_mode = "Animation View"
        min_cutoff = 1.0
        beta = 0.7

        if cv2.os.path.exists("background.jpg"): custom_bg = cv2.imread("background.jpg")
        else: custom_bg = None

        with mss.mss() as sct:
            monitor = self.capture_region if self.capture_region is not None else sct.monitors[1]
            monitor_width, monitor_height = monitor["width"], monitor["height"]

            if custom_bg is not None:
                custom_bg = cv2.resize(custom_bg, (monitor_width, monitor_height))

            while not self._stop_event.is_set():
                if not self.settings_queue.empty():
                    settings = self.settings_queue.get()
                    current_profile_name = settings.get("profile", current_profile_name)
                    current_bg_name = settings.get("background", current_bg_name)
                    current_confidence_threshold = settings.get("confidence_threshold", current_confidence_threshold)
                    current_view_mode = settings.get("view_mode", current_view_mode)
                    min_cutoff = settings.get("min_cutoff", min_cutoff)
                    beta = settings.get("beta", beta)
                    if "is_recording" in settings:
                        is_recording = settings["is_recording"]
                        # (此处省略了录制逻辑，因为它与当前任务无关)

                t0 = time.time()
                sct_img = sct.grab(monitor)
                img_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                results = self.model.track(img_bgr, persist=True, verbose=False, conf=current_confidence_threshold)

                if current_view_mode == "Debug View":
                    output_frame = results[0].plot()
                else:
                    if current_bg_name == "black": canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)
                    elif current_bg_name == "blue": canvas = np.full((monitor_height, monitor_width, 3), (139, 0, 0), dtype=np.uint8)
                    elif current_bg_name == "custom" and custom_bg is not None: canvas = custom_bg.copy()
                    else: canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)

                    track_ids = results[0].boxes.id
                    all_keypoints = results[0].keypoints.data.cpu().numpy()
                    current_frame_ids = set()

                    if track_ids is not None:
                        for i, person_keypoints in enumerate(all_keypoints):
                            track_id = int(track_ids[i])
                            current_frame_ids.add(track_id)

                            if track_id not in self.pose_filters:
                                print(f"后台线程: 检测到新的人物 (ID: {track_id}), 使用参数 min_cutoff={min_cutoff:.2f}, beta={beta:.2f} 初始化滤波器。")
                                self.pose_filters[track_id] = [OneEuroFilter(t0, p[0], min_cutoff=min_cutoff, beta=beta) for p in person_keypoints] + \
                                                              [OneEuroFilter(t0, p[1], min_cutoff=min_cutoff, beta=beta) for p in person_keypoints]

                            smoothed_kpts = np.zeros_like(person_keypoints)
                            num_kpts = len(person_keypoints)
                            for j in range(num_kpts):
                                smoothed_kpts[j, 0] = self.pose_filters[track_id][j](t0, person_keypoints[j, 0])
                                smoothed_kpts[j, 1] = self.pose_filters[track_id][j + num_kpts](t0, person_keypoints[j, 1])
                                smoothed_kpts[j, 2] = person_keypoints[j, 2]

                            draw_character(canvas, smoothed_kpts, CHARACTER_PROFILES[current_profile_name], confidence_threshold=current_confidence_threshold)

                    stale_ids = set(self.pose_filters.keys()) - current_frame_ids
                    for stale_id in stale_ids:
                        del self.pose_filters[stale_id]
                    output_frame = canvas

                draw_hud(output_frame, is_recording, current_profile_name, current_bg_name, current_view_mode)

                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(output_frame)

        print("后台线程：已停止。")
