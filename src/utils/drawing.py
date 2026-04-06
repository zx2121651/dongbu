import cv2
import numpy as np
from src.utils.config import BODY_PART_CONNECTIONS, KEYPOINT_DICT

def draw_character(canvas, keypoints, profile, confidence_threshold=0.5):
    for part_name, kp_indices in BODY_PART_CONNECTIONS.items():
        if part_name == "torso":
            pts = []
            valid_torso = True
            for idx in kp_indices:
                if keypoints[idx, 2] < confidence_threshold:
                    valid_torso = False
                    break
                pts.append([int(keypoints[idx, 0]), int(keypoints[idx, 1])])

            if valid_torso:
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(canvas, [pts], profile[part_name]["color"])
        else:
            idx1, idx2 = kp_indices
            if keypoints[idx1, 2] >= confidence_threshold and keypoints[idx2, 2] >= confidence_threshold:
                pt1 = (int(keypoints[idx1, 0]), int(keypoints[idx1, 1]))
                pt2 = (int(keypoints[idx2, 0]), int(keypoints[idx2, 1]))
                cv2.line(canvas, pt1, pt2, profile[part_name]["color"], profile[part_name]["thickness"])

    nose_idx = KEYPOINT_DICT["nose"]
    left_ear_idx = KEYPOINT_DICT["left_ear"]
    right_ear_idx = KEYPOINT_DICT["right_ear"]

    if keypoints[nose_idx, 2] >= confidence_threshold:
        center_x, center_y = int(keypoints[nose_idx, 0]), int(keypoints[nose_idx, 1])
        radius = 40
        if keypoints[left_ear_idx, 2] >= confidence_threshold and keypoints[right_ear_idx, 2] >= confidence_threshold:
             dist = np.sqrt((keypoints[left_ear_idx, 0] - keypoints[right_ear_idx, 0])**2 + (keypoints[left_ear_idx, 1] - keypoints[right_ear_idx, 1])**2)
             radius = int(dist * 0.8)
             radius = max(20, min(80, radius))
        cv2.circle(canvas, (center_x, center_y), radius, profile["head"]["color"], -1)
