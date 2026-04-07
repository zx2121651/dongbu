MODERN_DARK_STYLE = """
/* 全局基础设置 */
* {
    font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
    font-size: 14px;
    color: #E0E0E0;
}

QMainWindow, QWidget#central_widget {
    background-color: #1E1E24;
}

/* 滚动条美化 */
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollBar:vertical {
    background: #2C2C35;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #555560;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #707080;
}

/* 分组框 (类似面板卡片) */
QGroupBox {
    background-color: #262630;
    border: 1px solid #363640;
    border-radius: 8px;
    margin-top: 24px;
    padding-top: 16px;
    padding-bottom: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 4px;
    color: #A0A0B0;
    font-weight: bold;
    font-size: 13px;
}

/* 按钮样式 (仿国产现代化UI) */
QPushButton {
    background-color: #333340;
    border: 1px solid #444455;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #444455;
    border: 1px solid #555566;
}
QPushButton:pressed {
    background-color: #22222A;
}
QPushButton#primary_btn {
    background-color: #1890FF; /* Ant Design 蓝 */
    border: none;
    color: #FFFFFF;
}
QPushButton#primary_btn:hover {
    background-color: #40A9FF;
}
QPushButton#danger_btn {
    background-color: #FF4D4F;
    border: none;
    color: #FFFFFF;
}
QPushButton#danger_btn:hover {
    background-color: #FF7875;
}

/* 下拉菜单 */
QComboBox {
    background-color: #1E1E24;
    border: 1px solid #444455;
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 24px;
}
QComboBox:hover {
    border: 1px solid #1890FF;
}
QComboBox::drop-down {
    border: none;
    width: 24px;
}

/* 单选框与复选框 */
QRadioButton::indicator, QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 1px solid #555566;
    background-color: #1E1E24;
}
QCheckBox::indicator {
    border-radius: 4px;
}
QRadioButton::indicator:checked {
    background-color: #1890FF;
    border: 3px solid #1E1E24;
}
QCheckBox::indicator:checked {
    background-color: #1890FF;
}

/* 滑块 */
QSlider::groove:horizontal {
    border-radius: 3px;
    height: 6px;
    background: #333340;
}
QSlider::handle:horizontal {
    background: #1890FF;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    background: #40A9FF;
}

/* 标签 */
QLabel {
    color: #C0C0C0;
}
"""
