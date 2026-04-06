# -*- coding: utf-8 -*-

"""
主GUI应用程序模块 (PyQt6版)

该文件使用PyQt6创建主应用程序窗口，
并管理后台的捕捉线程。
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QScrollArea, QPushButton,
                             QComboBox, QRadioButton, QGroupBox, QSlider, QGridLayout, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
import cv2
import json
import os

from capture_thread import CaptureWorker
from config import CHARACTER_PROFILES, SAMPLE_KEYPOINTS
from drawing import draw_character
from roi_selector import ROISelector

class MainWindow(QMainWindow):
    # Signals to communicate with the worker thread
    view_mode_changed = pyqtSignal(str)
    profile_changed = pyqtSignal(str)
    background_changed = pyqtSignal(str)
    threshold_changed = pyqtSignal(float)
    min_cutoff_changed = pyqtSignal(float)
    beta_changed = pyqtSignal(float)
    recording_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("动画桌面捕捉 v4.0 (PyQt6)")
        self.setGeometry(100, 100, 1200, 720)

        self.capture_region = None
        self.selected_model = "yolo11n-pose.pt"
        self.is_recording = False

        self.settings_file = "settings.json"
        self.user_settings = self.load_settings()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.video_label = QLabel("正在启动...", self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 24px;")
        self.main_layout.addWidget(self.video_label, 1)

        self.create_settings_panel()
        self.start_capture_thread()
        self.update_preview(list(CHARACTER_PROFILES.keys())[0])

    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {
            "model": "yolo11n-pose.pt", "view_mode": "Animation View", "profile": list(CHARACTER_PROFILES.keys())[0],
            "background": "black", "threshold": 50, "min_cutoff": 100, "beta": 70
        }

    def save_settings(self):
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_settings, f, indent=4)

    def start_capture_thread(self):
        self.thread = QThread()
        self.worker = CaptureWorker(self.capture_region, self.selected_model)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.frame_ready.connect(self.update_frame)

        self.view_mode_changed.connect(self.worker.set_view_mode)
        self.profile_changed.connect(self.worker.set_profile)
        self.background_changed.connect(self.worker.set_background)
        self.threshold_changed.connect(self.worker.set_confidence_threshold)
        self.min_cutoff_changed.connect(self.worker.set_min_cutoff)
        self.beta_changed.connect(self.worker.set_beta)
        self.recording_changed.connect(self.worker.set_recording)

        self.thread.start()
        print(f"主GUI：捕捉线程已启动 (模型: {self.selected_model})。")

        # Emit initial values
        self.view_mode_changed.emit(self.user_settings.get("view_mode", "Animation View"))
        self.profile_changed.emit(self.user_settings.get("profile", list(CHARACTER_PROFILES.keys())[0]))
        self.background_changed.emit(self.user_settings.get("background", "black"))
        self.threshold_changed.emit(self.user_settings.get("threshold", 50) / 100.0)
        self.min_cutoff_changed.emit(self.user_settings.get("min_cutoff", 100) / 100.0)
        self.beta_changed.emit(self.user_settings.get("beta", 70) / 100.0)

    def create_settings_panel(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(300)

        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- Groups ---
        capture_group = QGroupBox("捕捉与模型"); appearance_group = QGroupBox("外观与视图"); params_group = QGroupBox("效果与参数")

        self.create_capture_group(capture_group)
        self.create_appearance_group(appearance_group)
        self.create_params_group(params_group)

        settings_layout.addWidget(capture_group)
        settings_layout.addWidget(appearance_group)
        settings_layout.addWidget(params_group)

        self.preview_label = QLabel()
        settings_layout.addWidget(self.preview_label)

        self.record_button = QPushButton("开始录制")
        self.record_button.clicked.connect(self.toggle_recording)
        settings_layout.addWidget(self.record_button)

        scroll_area.setWidget(settings_panel)
        self.main_layout.addWidget(scroll_area)

    def create_capture_group(self, group_box):
        layout = QVBoxLayout(group_box)
        roi_button = QPushButton("选择捕捉区域"); roi_button.clicked.connect(self.open_roi_selector)
        reset_roi_button = QPushButton("重置为全屏"); reset_roi_button.clicked.connect(self.reset_roi)
        model_combo = QComboBox(); model_combo.addItems(['yolo11n-pose.pt', 'yolo11s-pose.pt', 'yolo11m-pose.pt']); model_combo.currentTextChanged.connect(self.on_model_select)
        layout.addWidget(roi_button); layout.addWidget(reset_roi_button); layout.addWidget(model_combo)

    def create_appearance_group(self, group_box):
        layout = QVBoxLayout(group_box)
        self.view_mode_anim_radio = QRadioButton("动画视图"); self.view_mode_anim_radio.setChecked(True); self.view_mode_anim_radio.toggled.connect(lambda: self.on_view_mode_change("Animation View"))
        self.view_mode_debug_radio = QRadioButton("视频骨骼预览"); self.view_mode_debug_radio.toggled.connect(lambda: self.on_view_mode_change("Debug View"))
        profile_combo = QComboBox(); profile_combo.addItems(CHARACTER_PROFILES.keys())
        profile_combo.setCurrentText(self.user_settings.get("profile", list(CHARACTER_PROFILES.keys())[0]))
        profile_combo.currentTextChanged.connect(self.on_profile_select)

        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["black", "blue", "custom"])
        self.bg_combo.setCurrentText(self.user_settings.get("background", "black"))
        self.bg_combo.currentTextChanged.connect(self.on_background_select)

        self.custom_bg_btn = QPushButton("选择自定义背景")
        self.custom_bg_btn.clicked.connect(self.select_custom_bg)
        self.custom_bg_btn.setVisible(self.user_settings.get("background", "black") == "custom")

        layout.addWidget(self.view_mode_anim_radio); layout.addWidget(self.view_mode_debug_radio); layout.addWidget(profile_combo)
        layout.addWidget(QLabel("背景:"))
        layout.addWidget(self.bg_combo)
        layout.addWidget(self.custom_bg_btn)

    def create_params_group(self, group_box):
        layout = QGridLayout(group_box)
        def create_slider(label, min_val, max_val, initial_val, signal, settings_key):
            label_widget = QLabel(label)
            slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(min_val, max_val); slider.setValue(initial_val)
            value_label = QLabel(f"{initial_val/100.0:.2f}")
            def value_changed(v):
                value_label.setText(f"{v/100.0:.2f}")
                self.user_settings[settings_key] = v
                signal.emit(v/100.0)
            slider.valueChanged.connect(value_changed)
            row = layout.rowCount()
            layout.addWidget(label_widget, row, 0); layout.addWidget(slider, row, 1); layout.addWidget(value_label, row, 2)

        create_slider("置信度:", 0, 100, self.user_settings.get("threshold", 50), self.threshold_changed, "threshold")
        create_slider("Min Cutoff:", 10, 200, self.user_settings.get("min_cutoff", 100), self.min_cutoff_changed, "min_cutoff")
        create_slider("Beta:", 0, 150, self.user_settings.get("beta", 70), self.beta_changed, "beta")


    def on_view_mode_change(self, mode):
        if self.sender().isChecked():
            self.view_mode_changed.emit(mode)
            self.user_settings["view_mode"] = mode
    def on_profile_select(self, name):
        self.user_settings["profile"] = name
        self.profile_changed.emit(name); self.update_preview(name)

    def on_background_select(self, bg_name):
        self.user_settings["background"] = bg_name
        self.background_changed.emit(bg_name)
        self.custom_bg_btn.setVisible(bg_name == "custom")

    def select_custom_bg(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择背景图片", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            import shutil
            shutil.copy(file_name, "background.jpg")
            # Re-emit to trigger reload in thread
            self.background_changed.emit("custom")
    def on_model_select(self, name):
        if self.thread.isRunning(): self.worker.stop(); self.thread.quit(); self.thread.wait()
        self.selected_model = name; self.start_capture_thread()
    def open_roi_selector(self):
        self.hide(); self.roi_selector = ROISelector(self.roi_selection_callback); self.roi_selector.show()
    def reset_roi(self): self.roi_selection_callback(None)
    def roi_selection_callback(self, roi):
        if self.thread.isRunning(): self.worker.stop(); self.thread.quit(); self.thread.wait()
        self.capture_region = roi; self.show(); self.start_capture_thread()
    def toggle_recording(self):
        self.is_recording = not self.is_recording; self.recording_changed.emit(self.is_recording)
        if self.is_recording: self.record_button.setText("停止录制"); self.record_button.setStyleSheet("background-color: red;")
        else: self.record_button.setText("开始录制"); self.record_button.setStyleSheet("")

    def update_preview(self, profile_name):
        profile = CHARACTER_PROFILES[profile_name]; w, h = 150, 200
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        kpts = SAMPLE_KEYPOINTS.copy(); kpts[:, 0] *= w; kpts[:, 1] *= h
        draw_character(canvas, kpts, profile)
        self.preview_label.setPixmap(self.convert_cv_qt(canvas, (w, h)))

    @pyqtSlot(np.ndarray)
    def update_frame(self, cv_img):
        self.video_label.setPixmap(self.convert_cv_qt(cv_img, self.video_label.size()))
    def convert_cv_qt(self, img, size):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_img).scaled(size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    def closeEvent(self, event):
        self.save_settings()
        self.worker.stop(); self.thread.quit(); self.thread.wait(); event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
