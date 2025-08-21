# -*- coding: utf-8 -*-

"""
绘图模块

该文件包含了所有与OpenCV绘图相关的函数，
例如绘制肢体、绘制完整的角色，以及在屏幕上显示HUD信息。
"""

import cv2
import numpy as np
from config import BODY_PART_CONNECTIONS, KEYPOINT_DICT

def draw_limb(canvas, pt1, pt2, color, thickness):
    """在画布上绘制一个肢体（一个旋转的矩形）。"""
    length = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if length < 1: return
    angle = np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
    center = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
    rect = np.array([[-length / 2, -thickness / 2], [length / 2, -thickness / 2], [length / 2, thickness / 2], [-length / 2, thickness / 2]], dtype=np.float32)
    rot_mat = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    rotated_rect = cv2.transform(np.array([rect]), rot_mat)[0]
    translated_rect = rotated_rect + center
    cv2.fillConvexPoly(canvas, translated_rect.astype(np.int32), color)

def draw_character(canvas, keypoints, profile, confidence_threshold=0.5):
    """在给定的画布上根据关键点绘制一个更生动的角色。"""
    # 绘制躯干
    torso_indices = BODY_PART_CONNECTIONS["torso"]
    torso_points = [keypoints[i] for i in torso_indices]
    if all(p[2] > confidence_threshold for p in torso_points):
        pts = np.array([[p[0], p[1]] for p in torso_points], dtype=np.int32)
        cv2.fillConvexPoly(canvas, pts, profile["torso"]["color"])

    # 绘制四肢
    for part_name, indices in BODY_PART_CONNECTIONS.items():
        if part_name == "torso": continue
        kp1 = keypoints[indices[0]]
        kp2 = keypoints[indices[1]]
        if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
            pt1, pt2 = (kp1[0], kp1[1]), (kp2[0], kp2[1])
            # 从profile中获取相应部位的颜色和粗细
            part_profile = profile.get(part_name, {})
            color = part_profile.get("color", (255, 255, 255))
            thickness = part_profile.get("thickness", 5)
            draw_limb(canvas, pt1, pt2, color, thickness)

    # 绘制头部
    nose = keypoints[KEYPOINT_DICT["nose"]]
    if nose[2] > confidence_threshold:
        center = (int(nose[0]), int(nose[1]))
        ls = keypoints[KEYPOINT_DICT["left_shoulder"]]
        rs = keypoints[KEYPOINT_DICT["right_shoulder"]]
        if ls[2] > confidence_threshold and rs[2] > confidence_threshold:
            head_radius = int(np.hypot(ls[0] - rs[0], ls[1] - rs[1]) / 3)
            # 从profile中获取头部的颜色
            head_color = profile.get("head", {}).get("color", (255, 255, 255))
            cv2.circle(canvas, center, head_radius, head_color, -1)

def draw_hud(canvas, is_recording, profile_name, bg_name):
    """在屏幕上绘制HUD信息（状态和帮助）。"""
    # 录制状态
    rec_status = "REC" if is_recording else "IDLE"
    rec_color = (0, 0, 255) if is_recording else (0, 255, 0)
    cv2.putText(canvas, f"STATUS: {rec_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)

    # 其他信息
    info_text_1 = f"Profile: {profile_name} [c] | BG: {bg_name} [b]"
    info_text_2 = "Record: [r] | Quit: [q]"
    cv2.putText(canvas, info_text_1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, info_text_2, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
