import cv2
import mss
import numpy as np
from ultralytics import YOLO
import os
import time

# --- 数据定义 ---
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
    "default": { "torso": {"color": (255, 140, 0), "thickness": 10}, "left_upper_arm": {"color": (0, 255, 0), "thickness": 10}, "left_lower_arm": {"color": (0, 255, 0), "thickness": 8}, "right_upper_arm": {"color": (0, 0, 255), "thickness": 10}, "right_lower_arm": {"color": (0, 0, 255), "thickness": 8}, "left_upper_leg": {"color": (255, 255, 0), "thickness": 10}, "left_lower_leg": {"color": (255, 255, 0), "thickness": 8}, "right_upper_leg": {"color": (255, 0, 255), "thickness": 10}, "right_lower_leg": {"color": (255, 0, 255), "thickness": 8}, "head": {"color": (200, 200, 220)}},
    "monochrome": { "torso": {"color": (200, 200, 200), "thickness": 10}, "left_upper_arm": {"color": (150, 150, 150), "thickness": 10}, "left_lower_arm": {"color": (150, 150, 150), "thickness": 8}, "right_upper_arm": {"color": (150, 150, 150), "thickness": 10}, "right_lower_arm": {"color": (150, 150, 150), "thickness": 8}, "left_upper_leg": {"color": (180, 180, 180), "thickness": 10}, "left_lower_leg": {"color": (180, 180, 180), "thickness": 8}, "right_upper_leg": {"color": (180, 180, 180), "thickness": 10}, "right_lower_leg": {"color": (180, 180, 180), "thickness": 8}, "head": {"color": (255, 255, 255)}}
}

# --- 绘图函数 ---
def draw_limb(canvas, pt1, pt2, color, thickness):
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
    torso_indices = BODY_PART_CONNECTIONS["torso"]
    torso_points = [keypoints[i] for i in torso_indices]
    if all(p[2] > confidence_threshold for p in torso_points):
        pts = np.array([[p[0], p[1]] for p in torso_points], dtype=np.int32)
        cv2.fillConvexPoly(canvas, pts, profile["torso"]["color"])
    for part_name, indices in BODY_PART_CONNECTIONS.items():
        if part_name == "torso": continue
        kp1 = keypoints[indices[0]]
        kp2 = keypoints[indices[1]]
        if kp1[2] > confidence_threshold and kp2[2] > confidence_threshold:
            pt1, pt2 = (kp1[0], kp1[1]), (kp2[0], kp2[1])
            draw_limb(canvas, pt1, pt2, profile[part_name]["color"], profile[part_name]["thickness"])
    nose = keypoints[KEYPOINT_DICT["nose"]]
    if nose[2] > confidence_threshold:
        center = (int(nose[0]), int(nose[1]))
        ls = keypoints[KEYPOINT_DICT["left_shoulder"]]
        rs = keypoints[KEYPOINT_DICT["right_shoulder"]]
        if ls[2] > confidence_threshold and rs[2] > confidence_threshold:
            head_radius = int(np.hypot(ls[0] - rs[0], ls[1] - rs[1]) / 3)
            cv2.circle(canvas, center, head_radius, profile["head"]["color"], -1)

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


# --- 主程序 ---

def main():
    print("正在加载YOLOv8姿态估计模型...")
    model = YOLO("yolo11n-pose.pt")
    print("模型加载完成。查看窗口中的热键提示。")

    profile_names = list(CHARACTER_PROFILES.keys())
    current_profile_index = 0
    backgrounds = ["black", "blue"]
    current_bg_index = 0

    is_recording = False
    video_writer = None
    output_filename = "output.mp4"
    fps = 20.0

    if os.path.exists("background.jpg"):
        custom_bg = cv2.imread("background.jpg")
        backgrounds.append("custom")
    else:
        custom_bg = None
        print("提示: 未找到 background.jpg, 自定义背景将不可用。")

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        monitor_width, monitor_height = monitor["width"], monitor["height"]

        if custom_bg is not None:
            custom_bg = cv2.resize(custom_bg, (monitor_width, monitor_height))

        cv2.namedWindow("Animation Desktop Capture", cv2.WND_PROP_VISIBLE)

        while True:
            # --- 设置当前帧 ---
            current_profile_name = profile_names[current_profile_index]
            current_profile = CHARACTER_PROFILES[current_profile_name]
            bg_choice = backgrounds[current_bg_index]

            if bg_choice == "black": animation_canvas = np.zeros((monitor_height, monitor_width, 3), dtype=np.uint8)
            elif bg_choice == "blue": animation_canvas = np.full((monitor_height, monitor_width, 3), (139, 0, 0), dtype=np.uint8)
            elif bg_choice == "custom": animation_canvas = custom_bg.copy()

            # --- 核心逻辑 ---
            sct_img = sct.grab(monitor)
            img_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            results = model(img_bgr, stream=True, verbose=False)

            for r in results:
                for person_keypoints in r.keypoints.data.cpu().numpy():
                    draw_character(animation_canvas, person_keypoints, current_profile)

            # --- 绘制UI和写入视频 ---
            draw_hud(animation_canvas, is_recording, current_profile_name, bg_choice)
            cv2.imshow("Animation Desktop Capture", animation_canvas)
            if is_recording:
                video_writer.write(animation_canvas)

            # --- 按键处理 ---
            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == ord("q"): break
            elif key == ord("c"):
                current_profile_index = (current_profile_index + 1) % len(profile_names)
            elif key == ord("b"):
                current_bg_index = (current_bg_index + 1) % len(backgrounds)
            elif key == ord("r"):
                is_recording = not is_recording
                if is_recording:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (monitor_width, monitor_height))
                    print(f"开始录制... 保存到 {output_filename}")
                else:
                    if video_writer:
                        video_writer.release()
                    video_writer = None
                    print("停止录制。")

    if video_writer: video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
