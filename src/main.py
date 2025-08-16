import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import time

from gesture_recognition import AudioGeneration, HandTracker
from video_to_frame import VideoSource


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracker with Audio")

        self.camArr = []
        self.camNum = None
        self.after_id = None
        self.video_source = None

        # UI: camera dropdown
        self.selectCam = ttk.Combobox(self.root, state="readonly")
        self.selectCam.set("Select Camera")
        self.selectCam.bind("<<ComboboxSelected>>", self.on_camera_selected)
        self.selectCam.pack()

        # UI: Status Label
        self.status_label = ttk.Label(root, text="Initializing...")
        self.status_label.pack()

        # Canvas for video feed
        self.canvas = tk.Canvas(root, width=640, height=480, bg="black")
        self.canvas.pack()

        # Init tracker/audio classes
        self.hand_tracker = HandTracker()
        self.audio_gen = AudioGeneration(app_instance=self)

        # Buffers for smoothing
        self.frame_count = 0
        self.freq_buffer = []
        self.amp_buffer = []
        self.roomSize_buffer = []
        self.smoothing_window = 5
        self.update_interval = 5

        # Create a separate canvas for the reverb visualization
        self.visualize_canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.visualize_canvas.pack(side=tk.RIGHT)
        self.cube_image_tk = None

        self.find_cameras()

    def find_cameras(self, maxCameras=10):
        self.camArr = []
        print("Searching for available cameras...")
        for i in range(maxCameras):
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"Camera index {i:02d} OK!")
                    self.camArr.append(i)
                cap.release()

        self.selectCam['values'] = [str(cam) for cam in self.camArr]
        if self.camArr:
            self.camNum = self.camArr[0]
            self.selectCam.set(str(self.camNum))
            self.start_camera()
            self.status_label.config(text=f"Camera {self.camNum} selected and started.")
        else:
            self.selectCam.set("No Camera Found")
            self.selectCam['values'] = ["No Camera Found"]
            self.status_label.config(text="No camera found.")
            print("No camera found.")

    def on_camera_selected(self, event):
        selected = self.selectCam.get()
        if selected == "No Camera Found":
            self.status_label.config(text="No cameras available to select.")
            return

        try:
            new_cam = int(selected)
        except ValueError:
            self.status_label.config(text="Invalid camera selection.")
            return

        if new_cam != self.camNum:
            print(f"Switching to camera {new_cam}...")
            if self.after_id:
                self.root.after_cancel(self.after_id)
            self.camNum = new_cam
            self.start_camera()
            self.status_label.config(text=f"Switching to camera {self.camNum}...")

    def start_camera(self):
        if self.after_id:
            self.root.after_cancel(self.after_id)

        if self.video_source:
            self.video_source.release()

        self.video_source = VideoSource(self.camNum)
        if not self.video_source.is_opened():
            self.status_label.config(text=f"Failed to open camera {self.camNum}.")
            self.audio_gen.stop_audio()
            self.canvas.delete("all")
            return

        ret, frame = self.video_source.read_frame()
        if ret:
            height, width, _ = frame.shape
            self.canvas.config(width=width, height=height)
            self.update_video()

    def update_video(self):
        ret, frame = self.video_source.read_frame()
        if ret:
            frame = cv2.flip(frame, 1)
            results = self.hand_tracker.detect_hands(frame)

            saw_left, saw_right, freq, amplitude, room_size = self.hand_tracker.check_gestures(results, frame)

            if saw_left or saw_right:
                self.audio_gen.start_audio()
                self.frame_count += 1
                if self.frame_count % self.update_interval == 0:
                    self.freq_buffer.append(freq)
                    self.amp_buffer.append(amplitude)
                    self.roomSize_buffer.append(room_size)

                    if len(self.freq_buffer) > self.smoothing_window:
                        self.freq_buffer.pop(0)
                        self.amp_buffer.pop(0)
                        self.roomSize_buffer.pop(0)

                    avg_freq = sum(self.freq_buffer) / len(self.freq_buffer)
                    avg_amp = sum(self.amp_buffer) / len(self.amp_buffer)
                    avg_roomSize = sum(self.roomSize_buffer) / len(self.roomSize_buffer)

                    self.audio_gen.set_parameters(avg_freq, avg_amp, avg_roomSize)
            else:
                self.audio_gen.stop_audio()

            self.hand_tracker.draw_landmarks(frame, results)

            # Get the PIL Image from the audio_gen class
            if self.audio_gen.roomSize is not None:
                cube_image = self.audio_gen.visualizeReverb(self.audio_gen.roomSize)
                if cube_image:
                    self.cube_image_tk = ImageTk.PhotoImage(cube_image)
                    self.visualize_canvas.create_image(0, 0, anchor=tk.NW, image=self.cube_image_tk)

            img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.imgtk = img_tk

            self.after_id = self.root.after(10, self.update_video)
        else:
            self.status_label.config(text=f"Camera {self.camNum} disconnected.")
            if self.video_source:
                self.video_source.release()
            self.audio_gen.stop_audio()
            self.canvas.delete("all")
            self.after_id = self.root.after(1000, self.update_video)

    def on_close(self):
        self.audio_gen.stop_audio()
        if self.after_id:
            self.root.after_cancel(self.after_id)
        if self.video_source:
            self.video_source.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()