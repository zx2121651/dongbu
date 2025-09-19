# -*- coding: utf-8 -*-

"""
主GUI应用程序模块 (PyQt6版)

该文件使用PyQt6创建主应用程序窗口，
并管理后台的捕捉线程。
"""

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt, QThread, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
import cv2

from capture_thread import CaptureWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("动画桌面捕捉 v4.0 (PyQt6)")
        self.setGeometry(100, 100, 1200, 720)

        # --- 主布局 ---
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # --- 视频显示区域 ---
        self.video_label = QLabel("正在启动...", self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white; font-size: 24px;")

        self.main_layout.addWidget(self.video_label, 1)

        # --- 设置并启动后台线程 ---
        self.thread = QThread()
        self.worker = CaptureWorker() # Pass settings later
        self.worker.moveToThread(self.thread)

        # 连接信号和槽
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.frame_ready.connect(self.update_frame)

        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_frame(self, cv_img):
        """更新视频帧的槽函数。"""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """将OpenCV图像转换为Qt图像。"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        # 保持纵横比缩放
        return QPixmap.fromImage(convert_to_Qt_format).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

    def closeEvent(self, event):
        """处理窗口关闭事件。"""
        print("主GUI：正在关闭应用程序...")
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
