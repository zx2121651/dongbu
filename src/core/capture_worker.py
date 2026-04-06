import mss
import numpy as np
import cv2
import time
import os
from ultralytics import YOLO
import mediapipe as mp
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.utils.drawing import draw_character
from src.utils.config import CHARACTER_PROFILES
from src.utils.filters import OneEuroFilter
from src.core.recorder import VideoRecorder

class CaptureWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def __init__(self, capture_region=None, model_name="yolo11n-pose.pt"):
        super().__init__()
        self.capture_region = capture_region
        self.model_name = model_name
        self._is_running = True

        self.current_profile_name = list(CHARACTER_PROFILES.keys())[0]
        self.current_bg_name = "black"
        self.current_confidence_threshold = 0.5
        self.current_view_mode = "Animation View"
        self.min_cutoff = 1.0
        self.beta = 0.7
        self.is_recording = False

        self.custom_bg_reload_requested = False

        self.pose_filters = {}
        if "yolo" in self.model_name:
            self.model = YOLO(self.model_name)
            self.mp_pose = None
        elif self.model_name == "mediapipe-pose":
            self.model = None
            self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.recorder = None

    @pyqtSlot(str)
    def set_view_mode(self, mode): self.current_view_mode = mode
    @pyqtSlot(str)
    def set_profile(self, profile_name): self.current_profile_name = profile_name
    @pyqtSlot(str)
    def set_background(self, bg_name):
        self.current_bg_name = bg_name
        if bg_name == "custom":
            self.custom_bg_reload_requested = True
    @pyqtSlot(float)
    def set_confidence_threshold(self, threshold): self.current_confidence_threshold = threshold
    @pyqtSlot(float)
    def set_min_cutoff(self, min_cutoff):
        self.min_cutoff = min_cutoff
        self.pose_filters.clear()
    @pyqtSlot(float)
    def set_beta(self, beta):
        self.beta = beta
        self.pose_filters.clear()
    @pyqtSlot(bool)
    def set_recording(self, is_recording): self.is_recording = is_recording

    def run(self):
        custom_bg = None
        if os.path.exists("background.jpg"):
            custom_bg = cv2.imread("background.jpg")

        with mss.mss() as sct:
            monitor = self.capture_region if self.capture_region is not None else sct.monitors[1]
            monitor_width, monitor_height = monitor["width"], monitor["height"]

            if custom_bg is not None:
                custom_bg = cv2.resize(custom_bg, (monitor_width, monitor_height))

            self.recorder = VideoRecorder(resolution=(monitor_width, monitor_height))

            while self._is_running:
                if self.custom_bg_reload_requested:
                    if os.path.exists("background.jpg"):
                        new_bg = cv2.imread("background.jpg")
                        if new_bg is not None:
                            custom_bg = cv2.resize(new_bg, (monitor_width, monitor_height))
                    self.custom_bg_reload_requested = False

                if self.is_recording and not self.recorder.is_recording:
                    self.recorder.start()
                elif not self.is_recording and self.recorder.is_recording:
                    self.recorder.stop()

                t0 = time.time()
                sct_img = sct.grab(monitor)
                img_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

                track_ids = None
                all_keypoints = []
                output_frame = img_bgr.copy()

                if self.model is not None:
                    results = self.model.track(img_bgr, persist=True, verbose=False, conf=self.current_confidence_threshold)
                    if self.current_view_mode == "Debug View":
                        output_frame = results[0].plot()
                    if results[0].boxes.id is not None:
                        track_ids = results[0].boxes.id
                        all_keypoints = results[0].keypoints.data.cpu().numpy()
                elif self.mp_pose is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    results = self.mp_pose.process(img_rgb)

                    if self.current_view_mode == "Debug View":
                        if results.pose_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(output_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

                    if results.pose_landmarks:
                        track_ids = [0] # MP only supports single person by default

                        # Map MediaPipe landmarks to YOLO format
                        landmarks = results.pose_landmarks.landmark
                        mapped_kpts = np.zeros((17, 3), dtype=np.float32)

                        mp_to_yolo_map = {
                            0: 0,   # nose
                            2: 1,   # left eye
                            5: 2,   # right eye
                            7: 3,   # left ear
                            8: 4,   # right ear
                            11: 5,  # left shoulder
                            12: 6,  # right shoulder
                            13: 7,  # left elbow
                            14: 8,  # right elbow
                            15: 9,  # left wrist
                            16: 10, # right wrist
                            23: 11, # left hip
                            24: 12, # right hip
                            25: 13, # left knee
                            26: 14, # right knee
                            27: 15, # left ankle
                            28: 16  # right ankle
                        }

                        for mp_idx, yolo_idx in mp_to_yolo_map.items():
                            lm = landmarks[mp_idx]
                            mapped_kpts[yolo_idx] = [lm.x * monitor_width, lm.y * monitor_height, lm.visibility]

                        all_keypoints = [mapped_kpts]

                if self.current_view_mode != "Debug View":
                    if self.current_bg_name == "black": canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)
                    elif self.current_bg_name == "blue": canvas = np.full((monitor_height, monitor_width, 3), (139, 0, 0), dtype=np.uint8)
                    elif self.current_bg_name == "custom" and custom_bg is not None: canvas = custom_bg.copy()
                    else: canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)

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

                if self.is_recording:
                    cv2.putText(output_frame, "REC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.recorder.push_frame(output_frame)

                self.frame_ready.emit(output_frame)

        if self.recorder is not None:
            self.recorder.stop()
        self.finished.emit()

    def stop(self):
        self._is_running = False
