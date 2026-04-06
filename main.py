import sys
import json
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QScrollArea
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap

from src.core.capture_worker import CaptureWorker
from src.gui.components.settings_panel import SettingsPanel
from src.gui.components.preview_panel import PreviewPanel
from src.utils.config import CHARACTER_PROFILES

class MainWindow(QMainWindow):
    view_mode_changed = pyqtSignal(str)
    profile_changed = pyqtSignal(str)
    background_changed = pyqtSignal(str)
    threshold_changed = pyqtSignal(float)
    min_cutoff_changed = pyqtSignal(float)
    beta_changed = pyqtSignal(float)
    recording_changed = pyqtSignal(bool)
    export_motion_data_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 动画桌面捕捉 v2.0 (Componentized)")
        self.setGeometry(100, 100, 1200, 720)
        self.settings_file = "settings.json"
        self.user_settings = self.load_settings()
        self.is_recording = False
        self.thread = None
        self.worker = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)

        self.video_label = QLabel("正在启动...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 24px;")
        self.main_layout.addWidget(self.video_label, 1)

        self.setup_sidebar()
        self.start_thread()

    def load_settings(self):
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {"model": "yolo11n-pose.pt", "view_mode": "Animation View", "profile": list(CHARACTER_PROFILES.keys())[0],
                "background": "black", "threshold": 50, "min_cutoff": 100, "beta": 70}

    def save_settings(self):
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_settings, f, indent=4)

    def setup_sidebar(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(320)

        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.settings_panel = SettingsPanel(self, self.user_settings)
        self.preview_panel = PreviewPanel(self)
        self.preview_panel.update_preview(self.user_settings.get("profile", list(CHARACTER_PROFILES.keys())[0]))

        sidebar_layout.addWidget(self.settings_panel)
        sidebar_layout.addWidget(self.preview_panel)
        scroll_area.setWidget(sidebar_widget)
        self.main_layout.addWidget(scroll_area)

    def start_thread(self):
        if self.thread is not None and self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait()

        self.thread = QThread()
        self.worker = CaptureWorker(None, self.user_settings.get("model", "yolo11n-pose.pt"))
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
        self.export_motion_data_changed.connect(self.worker.set_export_motion_data)

        self.thread.start()

        # Emit current settings
        self.view_mode_changed.emit(self.user_settings.get("view_mode", "Animation View"))
        self.profile_changed.emit(self.user_settings.get("profile", list(CHARACTER_PROFILES.keys())[0]))
        self.background_changed.emit(self.user_settings.get("background", "black"))
        self.threshold_changed.emit(self.user_settings.get("threshold", 50) / 100.0)
        self.min_cutoff_changed.emit(self.user_settings.get("min_cutoff", 100) / 100.0)
        self.beta_changed.emit(self.user_settings.get("beta", 70) / 100.0)
        self.export_motion_data_changed.emit(self.user_settings.get("export_motion_data", False))

    # --- Callbacks ---
    def on_model_select(self, model_name):
        self.user_settings["model"] = model_name
        self.start_thread()

    def on_view_mode_change(self, mode):
        self.user_settings["view_mode"] = mode
        self.view_mode_changed.emit(mode)

    def on_profile_select(self, name):
        self.user_settings["profile"] = name
        self.profile_changed.emit(name)
        self.preview_panel.update_preview(name)

    def on_background_select(self, bg_name):
        self.user_settings["background"] = bg_name
        self.settings_panel.custom_bg_btn.setVisible(bg_name == "custom")
        self.background_changed.emit(bg_name)

    def on_export_data_toggle(self, checked):
        self.user_settings["export_motion_data"] = checked
        self.export_motion_data_changed.emit(checked)

    def toggle_recording(self):
        self.is_recording = not self.is_recording
        self.recording_changed.emit(self.is_recording)
        if self.is_recording:
            self.preview_panel.record_btn.setText("停止录制")
            self.preview_panel.record_btn.setStyleSheet("background-color: red; color: white;")
        else:
            self.preview_panel.record_btn.setText("开始录制")
            self.preview_panel.record_btn.setStyleSheet("")

    @pyqtSlot(np.ndarray)
    def update_frame(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.save_settings()
        self.worker.stop()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
