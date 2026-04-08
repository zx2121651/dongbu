import numpy as np
import cv2
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from src.utils.drawing import draw_character
from src.utils.config import CHARACTER_PROFILES, SAMPLE_KEYPOINTS

class PreviewPanel(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        layout = QVBoxLayout(self)

        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview_label)

        from PyQt6.QtWidgets import QCheckBox
        self.export_data_cb = QCheckBox("同时导出 3D 动作数据 (.json)")
        self.export_data_cb.setChecked(self.main.user_settings.get("export_motion_data", False))
        self.export_data_cb.toggled.connect(self.main.on_export_data_toggle)
        layout.addWidget(self.export_data_cb)

        self.record_btn = QPushButton("🔴 开始录制视频与动作")
        self.record_btn.setObjectName("primary_btn")
        self.record_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_btn.setMinimumHeight(40)
        self.record_btn.clicked.connect(self.main.toggle_recording)
        layout.addWidget(self.record_btn)

    def update_preview(self, profile_name):
        profile = CHARACTER_PROFILES[profile_name]
        w, h = 150, 200
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        kpts = SAMPLE_KEYPOINTS.copy()
        kpts[:, 0] *= w
        kpts[:, 1] *= h
        draw_character(canvas, kpts, profile)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        q_img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(pixmap)
