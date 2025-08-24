# -*- coding: utf-8 -*-

"""
ROI (Region of Interest) 选择器模块

该文件提供一个全屏、半透明的窗口，允许用户通过鼠标拖拽来选择一个屏幕区域。
选择完成后，它通过回调函数返回所选区域的坐标。
"""

import tkinter as tk

class ROISelector(tk.Toplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.master = master
        self.callback = callback
        self.start_x = None
        self.start_y = None
        self.rect = None

        # --- 窗口设置 ---
        # 获取屏幕尺寸
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()

        # 设置为无边框、置顶、半透明的全屏窗口
        self.geometry(f"{screen_width}x{screen_height}+0+0")
        self.overrideredirect(True)
        self.wm_attributes("-alpha", 0.3) # 设置透明度
        self.wm_attributes("-topmost", True) # 保持在最前

        # --- 画布设置 ---
        self.canvas = tk.Canvas(self, cursor="cross", bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- 绑定事件 ---
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Escape>", self.cancel) # 按下Esc键取消选择

    def on_button_press(self, event):
        """鼠标左键按下的事件处理"""
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # 如果已存在矩形，先删除
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = None

    def on_mouse_drag(self, event):
        """鼠标拖拽的事件处理"""
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        # 实时更新矩形预览
        if self.rect:
            self.canvas.delete(self.rect)

        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, cur_x, cur_y,
            outline='red', width=2
        )

    def on_button_release(self, event):
        """鼠标左键释放的事件处理"""
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

        # 确保坐标是从左上到右下
        left = min(self.start_x, end_x)
        top = min(self.start_y, end_y)
        right = max(self.start_x, end_x)
        bottom = max(self.start_y, end_y)

        # 准备mss兼容的坐标字典
        roi = {
            'left': int(left),
            'top': int(top),
            'width': int(right - left),
            'height': int(bottom - top)
        }

        # 通过回调函数返回结果，并销毁自身
        if roi['width'] > 0 and roi['height'] > 0:
            self.callback(roi)
        self.destroy()

    def cancel(self, event=None):
        """取消选择"""
        self.destroy()
