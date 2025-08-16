import cv2

class VideoSource:
    def __init__(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}.")
        else:
            preferred_width = 1280
            preferred_height = 720
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_height)

    def is_opened(self):
        return self.cap and self.cap.isOpened()

    def read_frame(self):
        if not self.is_opened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.is_opened():
            self.cap.release()