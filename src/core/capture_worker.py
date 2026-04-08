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
from src.core.motion_exporter import MotionExporter

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
        self.export_motion_data = False

        self.custom_bg_reload_requested = False

        self.pose_filters = {}
        self.mp_holistic = None
        if "yolo" in self.model_name:
            self.model = YOLO(self.model_name)
            self.mp_pose = None
        elif self.model_name == "mediapipe-pose":
            self.model = None
            self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        elif self.model_name == "mediapipe-holistic":
            self.model = None
            self.mp_pose = None
            self.mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
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
    @pyqtSlot(bool)
    def set_export_motion_data(self, export): self.export_motion_data = export

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
            self.motion_exporter = MotionExporter()
            frame_idx = 0

            from PyQt6.QtWidgets import QApplication
            while self._is_running:
                QApplication.processEvents()

                if self.custom_bg_reload_requested:
                    if os.path.exists("background.jpg"):
                        new_bg = cv2.imread("background.jpg")
                        if new_bg is not None:
                            custom_bg = cv2.resize(new_bg, (monitor_width, monitor_height))
                    self.custom_bg_reload_requested = False

                if self.is_recording and not self.recorder.is_recording:
                    self.recorder.start()
                    if self.export_motion_data:
                        self.motion_exporter.start()
                elif not self.is_recording and self.recorder.is_recording:
                    self.recorder.stop()
                    self.motion_exporter.stop()

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
                        kpts_yolo = results[0].keypoints.data.cpu().numpy()
                        # Pad with z=0.0 since YOLO 2D doesn't natively output Z
                        all_keypoints = np.zeros((kpts_yolo.shape[0], kpts_yolo.shape[1], 4), dtype=np.float32)
                        all_keypoints[:, :, :3] = kpts_yolo[:, :, :3]
                elif self.mp_pose is not None or self.mp_holistic is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    if self.mp_pose is not None:
                        results = self.mp_pose.process(img_rgb)
                    else:
                        results = self.mp_holistic.process(img_rgb)

                    if self.current_view_mode == "Debug View":
                        if getattr(results, 'pose_landmarks', None):
                            mp.solutions.drawing_utils.draw_landmarks(output_frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                        if getattr(results, 'face_landmarks', None):
                            mp.solutions.drawing_utils.draw_landmarks(output_frame, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
                                                                      mp.solutions.drawing_utils.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                                                      mp.solutions.drawing_utils.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
                        if getattr(results, 'left_hand_landmarks', None):
                            mp.solutions.drawing_utils.draw_landmarks(output_frame, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
                        if getattr(results, 'right_hand_landmarks', None):
                            mp.solutions.drawing_utils.draw_landmarks(output_frame, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

                    if getattr(results, 'pose_landmarks', None):
                        track_ids = [0] # MP only supports single person by default

                        # Map MediaPipe landmarks to YOLO format
                        landmarks = results.pose_landmarks.landmark
                        # Expand to shape (17, 4) -> x, y, vis, z
                        mapped_kpts = np.zeros((17, 4), dtype=np.float32)

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
                            mapped_kpts[yolo_idx] = [lm.x * monitor_width, lm.y * monitor_height, lm.visibility, lm.z * monitor_width]

                        # If holistic, extract hands (extend the 17 kpts to 17 + 21 + 21 = 59 kpts for 3D export)
                        if self.mp_holistic is not None:
                            hands_kpts = np.zeros((42, 4), dtype=np.float32)
                            if getattr(results, 'left_hand_landmarks', None):
                                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                                    hands_kpts[i] = [lm.x * monitor_width, lm.y * monitor_height, 1.0, lm.z * monitor_width]
                            if getattr(results, 'right_hand_landmarks', None):
                                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                                    hands_kpts[21 + i] = [lm.x * monitor_width, lm.y * monitor_height, 1.0, lm.z * monitor_width]
                            mapped_kpts = np.vstack((mapped_kpts, hands_kpts))

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
                                # Init filters for x, y, z (if z exists, else default 0)
                                self.pose_filters[track_id] = [OneEuroFilter(t0, p[0], min_cutoff=self.min_cutoff, beta=self.beta) for p in person_keypoints] + \
                                                              [OneEuroFilter(t0, p[1], min_cutoff=self.min_cutoff, beta=self.beta) for p in person_keypoints] + \
                                                              [OneEuroFilter(t0, p[3] if len(p) > 3 else 0.0, min_cutoff=self.min_cutoff, beta=self.beta) for p in person_keypoints]

                            smoothed_kpts = np.zeros_like(person_keypoints)
                            num_kpts = len(person_keypoints)
                            for j in range(num_kpts):
                                smoothed_kpts[j, 0] = self.pose_filters[track_id][j](t0, person_keypoints[j, 0])
                                smoothed_kpts[j, 1] = self.pose_filters[track_id][j + num_kpts](t0, person_keypoints[j, 1])
                                smoothed_kpts[j, 2] = person_keypoints[j, 2] # vis
                                if smoothed_kpts.shape[1] > 3 and len(person_keypoints[j]) > 3:
                                    smoothed_kpts[j, 3] = self.pose_filters[track_id][j + 2*num_kpts](t0, person_keypoints[j, 3])

                            draw_character(canvas, smoothed_kpts, CHARACTER_PROFILES[self.current_profile_name], confidence_threshold=self.current_confidence_threshold)

                    stale_ids = set(self.pose_filters.keys()) - current_frame_ids
                    for stale_id in stale_ids:
                        del self.pose_filters[stale_id]
                    output_frame = canvas

                if self.is_recording:
                    cv2.putText(output_frame, "REC", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.recorder.push_frame(output_frame)

                    if self.export_motion_data and track_ids is not None and len(all_keypoints) > 0:
                        # Extract the smoothed 3D keypoints from filter instead of raw
                        smoothed_all_kpts = []
                        for t_id in track_ids:
                            t_id = int(t_id)
                            if t_id in self.pose_filters:
                                num_pts = len(self.pose_filters[t_id]) // 3
                                skpts = np.zeros((num_pts, 4), dtype=np.float32)
                                for j in range(num_pts):
                                    skpts[j, 0] = self.pose_filters[t_id][j].x_prev
                                    skpts[j, 1] = self.pose_filters[t_id][j + num_pts].x_prev
                                    skpts[j, 2] = 1.0
                                    skpts[j, 3] = self.pose_filters[t_id][j + 2 * num_pts].x_prev
                                smoothed_all_kpts.append(skpts)
                        if smoothed_all_kpts:
                            self.motion_exporter.push_frame(frame_idx, track_ids, smoothed_all_kpts)

                frame_idx += 1

                self.frame_ready.emit(output_frame)

        if self.recorder is not None:
            self.recorder.stop()
        if hasattr(self, 'motion_exporter') and self.motion_exporter is not None:
            self.motion_exporter.stop()
        self.finished.emit()

    def stop(self):
        self._is_running = False
