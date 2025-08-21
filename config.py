# -*- coding: utf-8 -*-

"""
配置模块

该文件包含了应用程序的所有静态配置数据，
例如关键点定义、身体部位连接和角色外观配置文件。
将这些数据分离出来有助于保持主代码的整洁。
"""
import numpy as np

# COCO关键点映射 (0-indexed)
# 这是YOLOv8姿态估计模型输出关键点的标准顺序
KEYPOINT_DICT = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

# 身体部位的连接关系
# 定义了哪些关键点应该被连接起来形成一个身体部位
BODY_PART_CONNECTIONS = {
    "torso": [KEYPOINT_DICT["left_shoulder"], KEYPOINT_DICT["right_shoulder"], KEYPOINT_DICT["right_hip"], KEYPOINT_DICT["left_hip"]],
    "left_upper_arm": [KEYPOINT_DICT["left_shoulder"], KEYPOINT_DICT["left_elbow"]],
    "left_lower_arm": [KEYPOINT_DICT["left_elbow"], KEYPOINT_DICT["left_wrist"]],
    "right_upper_arm": [KEYPOINT_DICT["right_shoulder"], KEYPOINT_DICT["right_elbow"]],
    "right_lower_arm": [KEYPOINT_DICT["right_elbow"], KEYPOINT_DICT["right_wrist"]],
    "left_upper_leg": [KEYPOINT_DICT["left_hip"], KEYPOINT_DICT["left_knee"]],
    "left_lower_leg": [KEYPOINT_DICT["left_knee"], KEYPOINT_DICT["left_ankle"]],
    "right_upper_leg": [KEYPOINT_DICT["right_hip"], KEYPOINT_DICT["right_knee"]],
    "right_lower_leg": [KEYPOINT_DICT["right_knee"], KEYPOINT_DICT["right_ankle"]],
}

# 角色外观配置文件
# 定义了不同角色的外观（颜色和粗细）
CHARACTER_PROFILES = {
    "default": {
        "torso": {"color": (255, 140, 0), "thickness": 10},
        "left_upper_arm": {"color": (0, 255, 0), "thickness": 10},
        "left_lower_arm": {"color": (0, 255, 0), "thickness": 8},
        "right_upper_arm": {"color": (0, 0, 255), "thickness": 10},
        "right_lower_arm": {"color": (0, 0, 255), "thickness": 8},
        "left_upper_leg": {"color": (255, 255, 0), "thickness": 10},
        "left_lower_leg": {"color": (255, 255, 0), "thickness": 8},
        "right_upper_leg": {"color": (255, 0, 255), "thickness": 10},
        "right_lower_leg": {"color": (255, 0, 255), "thickness": 8},
        "head": {"color": (200, 200, 220)}
    },
    "monochrome": {
        "torso": {"color": (200, 200, 200), "thickness": 10},
        "left_upper_arm": {"color": (150, 150, 150), "thickness": 10},
        "left_lower_arm": {"color": (150, 150, 150), "thickness": 8},
        "right_upper_arm": {"color": (150, 150, 150), "thickness": 10},
        "right_lower_arm": {"color": (150, 150, 150), "thickness": 8},
        "left_upper_leg": {"color": (180, 180, 180), "thickness": 10},
        "left_lower_leg": {"color": (180, 180, 180), "thickness": 8},
        "right_upper_leg": {"color": (180, 180, 180), "thickness": 10},
        "right_lower_leg": {"color": (180, 180, 180), "thickness": 8},
        "head": {"color": (255, 255, 255)}
    }
}

# 用于预览的静态样本姿态
# 坐标是标准化的 (0-1)，置信度全部设为1
SAMPLE_KEYPOINTS = np.array([
    [0.5, 0.15, 1],  # nose
    [0.48, 0.12, 1], # left_eye
    [0.52, 0.12, 1], # right_eye
    [0.45, 0.15, 1], # left_ear
    [0.55, 0.15, 1], # right_ear
    [0.35, 0.25, 1], # left_shoulder
    [0.65, 0.25, 1], # right_shoulder
    [0.25, 0.4, 1],  # left_elbow
    [0.75, 0.4, 1],  # right_elbow
    [0.15, 0.55, 1], # left_wrist
    [0.85, 0.55, 1], # right_wrist
    [0.4, 0.55, 1],  # left_hip
    [0.6, 0.55, 1],  # right_hip
    [0.4, 0.75, 1],  # left_knee
    [0.6, 0.75, 1],  # right_knee
    [0.4, 0.95, 1],  # left_ankle
    [0.6, 0.95, 1],  # right_ankle
], dtype=np.float32)
