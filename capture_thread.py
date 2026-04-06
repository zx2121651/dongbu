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
import os
import time
import queue
import threading
from ultralytics import YOLO
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

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

        # --- Default Settings ---
        self.current_profile_name = list(CHARACTER_PROFILES.keys())[0]
        self.current_bg_name = "black"
        self.current_confidence_threshold = 0.5
        self.current_view_mode = "Animation View"
        self.min_cutoff = 1.0
        self.beta = 0.7
        self.is_recording = False
        self.custom_bg_reload_requested = False

        self.model = YOLO(self.model_name)
        self.pose_filters = {}

        # --- Recording Thread Variables ---
        self.record_queue = queue.Queue(maxsize=100)
        self.record_thread = None
        self.record_running = False
        print(f"工作线程：YOLOv8模型 {self.model_name} 已加载。")

    # --- Public Slots for settings ---
    @pyqtSlot(str)
    def set_view_mode(self, mode):
        self.current_view_mode = mode

    @pyqtSlot(str)
    def set_profile(self, profile_name):
        self.current_profile_name = profile_name

    @pyqtSlot(str)
    def set_background(self, bg_name):
        self.current_bg_name = bg_name
        self.custom_bg_reload_requested = True

    @pyqtSlot(float)
    def set_confidence_threshold(self, threshold):
        self.current_confidence_threshold = threshold

    @pyqtSlot(float)
    def set_min_cutoff(self, min_cutoff):
        self.min_cutoff = min_cutoff
        self.pose_filters.clear()

    @pyqtSlot(float)
    def set_beta(self, beta):
        self.beta = beta
        self.pose_filters.clear()

    @pyqtSlot(bool)
    def set_recording(self, is_recording):
        self.is_recording = is_recording

    def run(self):
        video_writer = None
        fps = 20.0

        if os.path.exists("background.jpg"):
            custom_bg = cv2.imread("background.jpg")
        else:
            custom_bg = None

        with mss.mss() as sct:
            monitor = self.capture_region if self.capture_region is not None else sct.monitors[1]
            monitor_width, monitor_height = monitor["width"], monitor["height"]

            if custom_bg is not None:
                custom_bg = cv2.resize(custom_bg, (monitor_width, monitor_height))

            while self._is_running:
                if self.custom_bg_reload_requested:
                    if os.path.exists("background.jpg"):
                        custom_bg = cv2.imread("background.jpg")
                        if custom_bg is not None:
                            custom_bg = cv2.resize(custom_bg, (monitor_width, monitor_height))
                    self.custom_bg_reload_requested = False

                # --- Video Writer Management ---
                if self.is_recording and not self.record_running:
                    self.record_running = True
                    self.record_thread = threading.Thread(target=self._record_worker, args=(fps, monitor_width, monitor_height))
                    self.record_thread.start()
                elif not self.is_recording and self.record_running:
                    self.record_running = False
                    if self.record_thread is not None:
                        self.record_thread.join()
                        self.record_thread = None

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

                if self.is_recording and self.record_running:
                    if not self.record_queue.full():
                        self.record_queue.put(output_frame.copy())

                self.frame_ready.emit(output_frame)

        self.record_running = False
        if self.record_thread is not None:
            self.record_thread.join()
        self.finished.emit()
        print("工作线程：已停止。")

    def _record_worker(self, fps, monitor_width, monitor_height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter("output.mp4", fourcc, fps, (monitor_width, monitor_height))
        print("录像线程：开始录制...")

        while self.record_running or not self.record_queue.empty():
            try:
                frame = self.record_queue.get(timeout=0.5)
                video_writer.write(frame)
            except queue.Empty:
                continue

        video_writer.release()
        print("录像线程：停止录制。")

    def stop(self):
        self._is_running = False
        self.record_running = False
        if self.record_thread is not None:
            self.record_thread.join()
