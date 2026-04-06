import os
import shutil
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QComboBox,
                             QPushButton, QRadioButton, QSlider, QLabel, QGridLayout, QFileDialog)
from PyQt6.QtCore import Qt

class SettingsPanel(QWidget):
    def __init__(self, main_window, settings):
        super().__init__()
        self.main = main_window
        self.settings = settings
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        capture_group = QGroupBox("捕捉与模型")
        appearance_group = QGroupBox("外观与视图")
        params_group = QGroupBox("效果与参数")

        self.create_capture_group(capture_group)
        self.create_appearance_group(appearance_group)
        self.create_params_group(params_group)

        layout.addWidget(capture_group)
        layout.addWidget(appearance_group)
        layout.addWidget(params_group)

    def create_capture_group(self, group):
        layout = QVBoxLayout(group)
        model_combo = QComboBox()
        model_combo.addItems([
            'yolo11n-pose.pt', 'yolo11s-pose.pt', 'yolo11m-pose.pt',
            'yolo26n-pose.pt', 'yolo26s-pose.pt', 'yolo26m-pose.pt',
            'mediapipe-pose'
        ])
        model_combo.setCurrentText(self.settings.get("model", "yolo11n-pose.pt"))
        model_combo.currentTextChanged.connect(self.main.on_model_select)
        layout.addWidget(QLabel("模型:"))
        layout.addWidget(model_combo)

    def create_appearance_group(self, group):
        from src.utils.config import CHARACTER_PROFILES
        layout = QVBoxLayout(group)

        self.anim_radio = QRadioButton("动画视图")
        self.debug_radio = QRadioButton("视频骨骼预览")
        is_anim = self.settings.get("view_mode", "Animation View") == "Animation View"
        self.anim_radio.setChecked(is_anim)
        self.debug_radio.setChecked(not is_anim)
        self.anim_radio.toggled.connect(lambda: self.main.on_view_mode_change("Animation View") if self.anim_radio.isChecked() else None)
        self.debug_radio.toggled.connect(lambda: self.main.on_view_mode_change("Debug View") if self.debug_radio.isChecked() else None)

        profile_combo = QComboBox()
        profile_combo.addItems(list(CHARACTER_PROFILES.keys()))
        profile_combo.setCurrentText(self.settings.get("profile", list(CHARACTER_PROFILES.keys())[0]))
        profile_combo.currentTextChanged.connect(self.main.on_profile_select)

        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["black", "blue", "custom"])
        self.bg_combo.setCurrentText(self.settings.get("background", "black"))
        self.bg_combo.currentTextChanged.connect(self.main.on_background_select)

        self.custom_bg_btn = QPushButton("选择自定义背景")
        self.custom_bg_btn.setVisible(self.bg_combo.currentText() == "custom")
        self.custom_bg_btn.clicked.connect(self.select_custom_bg)

        layout.addWidget(self.anim_radio)
        layout.addWidget(self.debug_radio)
        layout.addWidget(QLabel("角色配置:"))
        layout.addWidget(profile_combo)
        layout.addWidget(QLabel("背景:"))
        layout.addWidget(self.bg_combo)
        layout.addWidget(self.custom_bg_btn)

    def create_params_group(self, group):
        layout = QGridLayout(group)
        def create_slider(label, min_val, max_val, initial_val, signal, key):
            label_widget = QLabel(label)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(initial_val)
            value_label = QLabel(f"{initial_val/100.0:.2f}")
            def val_changed(v):
                value_label.setText(f"{v/100.0:.2f}")
                self.settings[key] = v
                signal.emit(v/100.0)
            slider.valueChanged.connect(val_changed)
            row = layout.rowCount()
            layout.addWidget(label_widget, row, 0)
            layout.addWidget(slider, row, 1)
            layout.addWidget(value_label, row, 2)

        create_slider("置信度:", 0, 100, self.settings.get("threshold", 50), self.main.threshold_changed, "threshold")
        create_slider("Cutoff:", 10, 200, self.settings.get("min_cutoff", 100), self.main.min_cutoff_changed, "min_cutoff")
        create_slider("Beta:", 0, 150, self.settings.get("beta", 70), self.main.beta_changed, "beta")

    def select_custom_bg(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择背景图片", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            shutil.copy(file_name, "background.jpg")
            self.main.background_changed.emit("custom")
