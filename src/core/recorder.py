import cv2
import threading
import queue

class VideoRecorder:
    def __init__(self, filename="output.mp4", fps=20.0, resolution=(1920, 1080)):
        self.filename = filename
        self.fps = fps
        self.resolution = resolution
        self.queue = queue.Queue(maxsize=100)
        self.is_recording = False
        self.thread = None
        self.writer = None

    def start(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.thread = threading.Thread(target=self._record_loop)
        self.thread.start()

    def stop(self):
        self.is_recording = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def push_frame(self, frame):
        if self.is_recording and not self.queue.full():
            self.queue.put(frame.copy())

    def _record_loop(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, self.resolution)

        while self.is_recording or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.5)
                self.writer.write(frame)
            except queue.Empty:
                continue

        self.writer.release()
