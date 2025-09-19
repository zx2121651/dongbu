# -*- coding: utf-8 -*-

"""
捕捉与处理线程模块 (PyQt6版)

该文件定义了在后台运行的核心工作QObject。
它负责屏幕捕捉、姿态估计和图像绘制，
并通过PyQt的信号机制将最终的图像帧发送出去。
"""

import mss
import numpy as np
import cv2
import time
from ultralytics import YOLO
from PyQt6.QtCore import QObject, pyqtSignal

from drawing import draw_character, draw_hud
from config import CHARACTER_PROFILES
from one_euro_filter import OneEuroFilter

class CaptureWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def __init__(self, capture_region=None, model_name="yolo11n-pose.pt"):
        super().__init__()
        self.capture_region = capture_region
        self.model_name = model_name
        self._is_running = True

        # --- Settings ---
        self.current_profile_name = list(CHARACTER_PROFILES.keys())[0]
        self.current_bg_name = "black"
        self.current_confidence_threshold = 0.5
        self.current_view_mode = "Animation View"
        self.min_cutoff = 1.0
        self.beta = 0.7
        self.is_recording = False

        self.model = YOLO(self.model_name)
        self.pose_filters = {}
        print(f"工作线程：YOLOv8模型 {self.model_name} 已加载。")

    def run(self):
        """线程的主执行循环。"""
        if cv2.os.path.exists("background.jpg"):
            custom_bg = cv2.imread("background.jpg")
        else:
            custom_bg = None

        with mss.mss() as sct:
            monitor = self.capture_region if self.capture_region is not None else sct.monitors[1]
            monitor_width, monitor_height = monitor["width"], monitor["height"]

            if custom_bg is not None:
                custom_bg = cv2.resize(custom_bg, (monitor_width, monitor_height))

            while self._is_running:
                t0 = time.time()
                sct_img = sct.grab(monitor)
                img_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
                results = self.model.track(img_bgr, persist=True, verbose=False, conf=self.current_confidence_threshold)

                if self.current_view_mode == "Debug View":
                    output_frame = results[0].plot()
                else:
                    if self.current_bg_name == "black": canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)
                    elif self.current_bg_name == "blue": canvas = np.full((monitor_height, monitor_width, 3), (139, 0, 0), dtype=np.uint8)
                    elif self.current_bg_name == "custom" and custom_bg is not None: canvas = custom_bg.copy()
                    else: canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)

                    track_ids = results[0].boxes.id
                    all_keypoints = results[0].keypoints.data.cpu().numpy()
                    current_frame_ids = set()

                    if track_ids is not None:
                        for i, person_keypoints in enumerate(all_keypoints):
                            track_id = int(track_ids[i])
                            current_frame_ids.add(track_id)

                            if track_id not in self.pose_filters:
                                self.pose_filters[track_id] = [OneEuroFilter(t0, p[0], min_cutoff=self.min_cutoff, beta=self.beta) for p in person_keypoints] + \
                                                              [OneEuroFilter(t0, p[1], min_cutoff=self.min_cutoff, beta=self.beta) for p in person_keypoints]

                            smoothed_kpts = np.zeros_like(person_keypoints)
                            num_kpts = len(person_keypoints)
                            for j in range(num_kpts):
                                smoothed_kpts[j, 0] = self.pose_filters[track_id][j](t0, person_keypoints[j, 0])
                                smoothed_kpts[j, 1] = self.pose_filters[track_id][j + num_kpts](t0, person_keypoints[j, 1])
                                smoothed_kpts[j, 2] = person_keypoints[j, 2]

                            draw_character(canvas, smoothed_kpts, CHARACTER_PROFILES[self.current_profile_name], confidence_threshold=self.current_confidence_threshold)

                    stale_ids = set(self.pose_filters.keys()) - current_frame_ids
                    for stale_id in stale_ids:
                        del self.pose_filters[stale_id]
                    output_frame = canvas

                draw_hud(output_frame, self.is_recording, self.current_profile_name, self.current_bg_name, self.current_view_mode)

                self.frame_ready.emit(output_frame)

        self.finished.emit()
        print("工作线程：已停止。")

    def stop(self):
        self._is_running = False
