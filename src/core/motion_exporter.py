import json
import time
import os
import threading
import queue

class MotionExporter:
    def __init__(self, fps=20.0):
        self.fps = fps
        self.is_recording = False
        self.frames_data = []
        self.filename = None
        self.start_time = 0

    def start(self, filename="motion_data.json"):
        if self.is_recording: return
        self.filename = filename
        self.frames_data = []
        self.is_recording = True
        self.start_time = time.time()
        print(f"Motion Exporter: 开始记录动作数据到 {self.filename}")

    def push_frame(self, frame_index, track_ids, keypoints):
        if not self.is_recording: return

        timestamp = time.time() - self.start_time

        frame_record = {
            "frame": frame_index,
            "timestamp": timestamp,
            "tracks": []
        }

        for idx, t_id in enumerate(track_ids):
            kpts = keypoints[idx]
            # Convert to list for JSON serialization: [x, y, vis, z]
            kp_list = kpts.tolist()
            frame_record["tracks"].append({
                "id": int(t_id),
                "keypoints": kp_list
            })

        self.frames_data.append(frame_record)

    def stop(self):
        if not self.is_recording: return
        self.is_recording = False

        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "fps": self.fps,
                        "format": "yolo_coco_17_plus_z",
                        "duration": time.time() - self.start_time,
                        "frames_count": len(self.frames_data)
                    },
                    "frames": self.frames_data
                }, f)
            print(f"Motion Exporter: 动作数据已保存至 {self.filename}")
        except Exception as e:
            print(f"Motion Exporter Error: 保存失败 - {e}")
        finally:
            self.frames_data = []
