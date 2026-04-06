import numpy as np

KEYPOINT_DICT = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

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

SAMPLE_KEYPOINTS = np.array([
    [0.5, 0.15, 1], [0.48, 0.12, 1], [0.52, 0.12, 1], [0.45, 0.15, 1], [0.55, 0.15, 1],
    [0.35, 0.25, 1], [0.65, 0.25, 1], [0.25, 0.4, 1], [0.75, 0.4, 1], [0.15, 0.55, 1],
    [0.85, 0.55, 1], [0.4, 0.55, 1], [0.6, 0.55, 1], [0.4, 0.75, 1], [0.6, 0.75, 1],
    [0.4, 0.95, 1], [0.6, 0.95, 1],
], dtype=np.float32)
