import time

import cv2
import mediapipe as mp
import threading
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import math
from itertools import product
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing_utils = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results


class AudioGeneration:
    def __init__(self, app_instance=None):
        self.fs = 16000  # Sample rate
        self.freq = 440.0
        self.amplitude = 0.2
        self.running = False
        self.lock = threading.Lock()
        self.phase = 0.0
        self.roomSize = 0.3  # Affects reverb time
        self.phase_offset = 0
        self.app_instance = app_instance

    def start_audio(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.audioGen, daemon=True).start()

    def stop_audio(self):
        with self.lock:
            self.running = False

    def set_frequency(self, freq):
        with self.lock:
            self.freq = freq

    def set_amplitude(self, amplitude):
        with self.lock:
            self.amplitude = amplitude

    def set_room_size(self, roomSize):
        with self.lock:
            self.roomSize = roomSize
            if self.app_instance:  # check if app_instance exists
                self.app_instance.visualizeReverb(self.roomSize)

    def set_parameters(self, freq, amplitude, roomSize):
        with self.lock:
            alpha = 0.2  # Smoothing factor
            self.freq = self.freq * (1 - alpha) + freq * alpha
            self.amplitude = self.amplitude * (1 - alpha) + amplitude * alpha
            self.roomSize = self.roomSize * (1 - alpha) + roomSize * alpha
            if self.app_instance:
                self.app_instance.visualizeReverb(self.roomSize)

    def apply_reverb(self, signal, reverb_time, decay_factor=0.6, num_echoes=5):

        reverb_signal = np.copy(signal)
        delay_samples = int((reverb_time / num_echoes) * self.fs)

        for i in range(1, num_echoes + 1):
            decay = decay_factor ** i
            echo = np.zeros_like(signal)
            if delay_samples * i < len(signal):
                echo[delay_samples * i:] = signal[:-delay_samples * i] * decay
                reverb_signal += echo

        return np.clip(reverb_signal, -1.0, 1.0)

    def audioGen(self):
        def callback(outdata, frames, time, status):
            with self.lock:
                if not self.running:
                    raise sd.CallbackStop()

                t = (np.arange(frames) + self.phase_offset) / self.fs
                samples = self.amplitude * np.sin(2 * np.pi * self.freq * t).astype(np.float32)

                # Apply reverb using roomSize as the reverb time
                samples = self.apply_reverb(samples, self.roomSize)

                outdata[:, 0] = samples
                self.phase_offset += frames
                self.phase_offset %= self.fs

        try:
            with sd.OutputStream(channels=1, callback=callback, samplerate=self.fs, dtype='float32', blocksize=1024):
                while True:
                    with self.lock:
                        if not self.running:
                            break
                    sd.sleep(100)
        except Exception as e:
            print(f"Audio stream error: {e}")
            with self.lock:
                self.running = False


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracker with Audio")

        self.camArr = []
        self.camNum = None
        self.cap = None
        self.after_id = None  # To store the ID of the scheduled 'after' call

        # UI: camera dropdown
        self.selectCam = ttk.Combobox(self.root, state="readonly")
        self.selectCam.set("Select Camera")
        self.selectCam.bind("<<ComboboxSelected>>", self.on_camera_selected)
        self.selectCam.pack()

        # UI: Status Label
        self.status_label = ttk.Label(root, text="Initializing...")
        self.status_label.pack()

        # Initialize canvas with a default small size. It will be resized later.
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

        #init matplotlib
        # Matplotlib plot for reverb visualization
        self.fig = plt.Figure(figsize=(3, 3), dpi=100, facecolor='None') #facecolor = 'none' sets the figure background transparent, (3,3) creates a smaller figure
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Reverb Room Size")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas_plot.get_tk_widget()
        #self.canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True) # Commented out as per your previous request

        self.visualizeReverb(self.audio_gen.roomSize)
        ##init cube_image to none, updated through visualizeReverb
        self.cube_image = None
        self.cube_image_tk = None

        # Detect available cameras and populate dropdown
        self.select_camera(maxCameras=10) #calls update_video

    def select_camera(self, maxCameras=10):
        self.camArr = []
        for i in range(maxCameras):
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)  # MSMF backend
            if not cap.isOpened():
                cap.release()
                continue

            # Attempt to set a temporary small resolution for a quick check
            # This is just for detection, not for the final display
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

            ret, frame = cap.read()
            # Check if frame is not None, has data, and isn't mostly black
            if ret and frame is not None and frame.size > 0 and frame.sum() > (frame.size * 10):  # Increased threshold
                print(f"Camera index {i:02d} OK!")
                self.camArr.append(i)
            else:
                print(f"Camera index {i:02d} found but black screen or no data.")
            cap.release()  # Always release the capture object

        self.selectCam['values'] = [str(cam) for cam in self.camArr]

        # Set the default selection if cameras are found and auto-start
        if self.camArr:
            self.camNum = self.camArr[0]
            self.selectCam.set(str(self.camNum))
            self.start_camera()  # Auto-start the first available camera
            self.status_label.config(text=f"Camera {self.camNum} selected and started.")
        else:
            self.selectCam.set("No Camera Found")
            self.selectCam['values'] = ["No Camera Found"]  # Prevents user from trying to select "Select Camera"
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
            print("Invalid camera selection.")
            return

        if new_cam in self.camArr:
            if new_cam != self.camNum:
                print(f"Switching to camera {new_cam}...")
                # Stop current video processing loop before switching
                if self.after_id:
                    self.root.after_cancel(self.after_id)
                    self.after_id = None

                self.camNum = new_cam
                self.start_camera()
                self.status_label.config(text=f"Switching to camera {self.camNum}...")
            else:
                self.status_label.config(text=f"Camera {new_cam} is already selected.")
                print(f"Camera {new_cam} is already selected.")
        else:
            self.status_label.config(text=f"Camera {new_cam} is not available.")
            print(f"Camera {new_cam} is not available.")

    def start_camera(self):
        # Stop any currently running video loop first
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        # Release the current camera if it's open
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None  # Explicitly set to None after releasing

        self.cap = cv2.VideoCapture(self.camNum, cv2.CAP_MSMF)

        if not self.cap.isOpened():
            self.status_label.config(text=f"Failed to open camera {self.camNum}. Please check connection.")
            print(f"Failed to open camera {self.camNum}")
            self.cap = None  # Ensure it's None if opening failed
            self.audio_gen.stop_audio()  # Stop audio if camera fails
            self.canvas.delete("all")  # Clear canvas
            return

        # Attempt to set a preferred resolution, but it's not guaranteed.
        # This can help guide the camera to a common resolution if it supports it.
        # Otherwise, we'll use whatever resolution it provides.
        preferred_width = 1280
        preferred_height = 720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, preferred_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preferred_height)

        # Read a frame to get the actual resolution the camera opened with
        ret, frame = self.cap.read()

        if not ret or frame is None or frame.size == 0 or frame.sum() < (
                frame.size * 10):  # Increased threshold for black frame
            self.status_label.config(text=f"Camera {self.camNum} opened but shows black screen/no data. Releasing...")
            print(f"Camera {self.camNum} opened but black screen or no data.")
            self.cap.release()
            self.cap = None  # Ensure it's None if it's a black screen
            self.audio_gen.stop_audio()  # Stop audio if camera fails
            self.canvas.delete("all")  # Clear canvas
            return

        # Get actual dimensions from the first valid frame
        actual_height, actual_width, _ = frame.shape

        # Resize the canvas to match the actual camera resolution
        self.canvas.config(width=actual_width, height=actual_height)
        self.canvas.delete("all")  # Clear anything old on the canvas after resize

        print(f"Camera {self.camNum} started successfully with resolution {actual_width}x{actual_height}.")
        self.status_label.config(text=f"Camera {self.camNum} is active.")
        # Pass the first frame to update_video to avoid re-reading
        self.update_video(initial_frame=(ret, frame))  # Pass both ret and frame

    def update_video(self, initial_frame=None):
        # If no camera is active, clear canvas and update status
        if self.cap is None or not self.cap.isOpened():
            self.canvas.delete("all")  # Clear the canvas
            self.status_label.config(text="Waiting for camera selection or connection...")
            self.audio_gen.stop_audio()  # Ensure audio is stopped if camera is not active
            self.after_id = self.root.after(1000, self.update_video)  # Re-check after a short delay
            return

        # Initialize ret and frame
        ret = False
        frame = None

        if initial_frame is not None:
            ret, frame = initial_frame  # Unpack ret and frame
            initial_frame = None # Clear initial frame so subsequent calls read from camera
        else:
            ret, frame = self.cap.read()

        if ret:  # Now 'ret' is guaranteed to be assigned
            frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect

            # Call detect_hands and pass the frame
            results = self.hand_tracker.detect_hands(frame)

            # Pass the results and frame to check_gesture
            self.check_gesture(results, frame)

            if self.cube_image is not None:
                # Resize cube image to desired overlay size (e.g., 200x200 pixels)
                overlay_size = 200
                cube_resized = self.cube_image.resize((overlay_size, overlay_size), Image.LANCZOS)
                cube_np = np.array(cube_resized)  # Convert back to numpy array (RGBA)

                # Convert video frame to RGBA for alpha blending
                frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

                # Define position for the overlay (e.g., top-left corner with some padding)
                x_offset = 20
                y_offset = 20

                # Check bounds to ensure overlay fits within the frame
                y1, y2 = y_offset, y_offset + cube_np.shape[0]
                x1, x2 = x_offset, x_offset + cube_np.shape[1]

                # Ensure dimensions match for blending
                alpha_s = cube_np[:, :, 3] / 255.0  # Alpha channel of the cube
                alpha_l = 1.0 - alpha_s  # Inverse alpha

                for c in range(0, 3):  # Iterate over R, G, B channels
                    frame_rgba[y1:y2, x1:x2, c] = (alpha_s * cube_np[:, :, c] +
                                                   alpha_l * frame_rgba[y1:y2, x1:x2, c])

                frame = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

            # Convert frame to PhotoImage and display on canvas
            img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.imgtk = img_tk  # Keep a reference to prevent garbage collection

            # Schedule next update AFTER the current frame is processed
            self.after_id = self.root.after(10, self.update_video)
        else:
            # If camera stops returning frames (e.g., disconnected or error)
            print("Failed to read frame from camera. Releasing camera.")
            self.status_label.config(text=f"Camera {self.camNum} disconnected or failed to read frames.")
            if self.cap:
                self.cap.release()
                self.cap = None  # Indicate no active camera
            self.audio_gen.stop_audio()
            self.canvas.delete("all")  # Clear the display
            self.after_id = self.root.after(1000, self.update_video)# Try to update again after a second

    def check_gesture(self, results, frame):
        saw_left = False
        saw_right = False

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label

                self.hand_tracker.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )

                if handedness == "Left":
                    saw_left = True
                    Lpalm_pos = hand_landmarks.landmark[0]

                    # If you want low-to-high frequency to map to visual right-to-left hand movement (more natural for a mirror):
                    freq = 220 + (880 - 220) * (1.0 - Lpalm_pos.x)  # Inverted X for frequency

                    amplitude = 0.1 + 0.4 * (
                            1.0 - Lpalm_pos.y)  # Inverted Y for amplitude (higher on screen = louder)

                    self.freq_buffer.append(freq)
                    self.amp_buffer.append(amplitude)
                    if len(self.freq_buffer) > self.smoothing_window:
                        self.freq_buffer.pop(0)
                        self.amp_buffer.pop(0)



                if handedness == "Right":
                    saw_right = True
                    thumbTip_pos = hand_landmarks.landmark[4]
                    indexTip_pos = hand_landmarks.landmark[8]

                    thumbTip_posX = thumbTip_pos.x
                    thumbTip_posY = thumbTip_pos.y
                    indexTip_posX = indexTip_pos.x
                    indexTip_posY = indexTip_pos.y

                    tipDistance_reverb = math.hypot(
                        thumbTip_posX - indexTip_posX, thumbTip_posY - indexTip_posY
                    )

                    min_dist = 0.03
                    max_dist = 0.4
                    normalized_dist = (tipDistance_reverb - min_dist) / (max_dist - min_dist)
                    normalized_dist = min(max(normalized_dist, 0.0), 1.0)

                    room_size = 0.1 + normalized_dist * 0.9

                    self.roomSize_buffer.append(room_size)
                    if len(self.roomSize_buffer) > self.smoothing_window:
                        self.roomSize_buffer.pop(0)

            self.audio_gen.start_audio()

            self.frame_count += 1
            if self.frame_count % self.update_interval == 0 and (saw_left or saw_right):
                avg_freq = sum(self.freq_buffer) / len(
                    self.freq_buffer) if self.freq_buffer else self.audio_gen.freq
                avg_amp = sum(self.amp_buffer) / len(
                    self.amp_buffer) if self.amp_buffer else self.audio_gen.amplitude
                avg_roomSize = sum(self.roomSize_buffer) / len(
                    self.roomSize_buffer) if self.roomSize_buffer else self.audio_gen.roomSize

                self.audio_gen.set_parameters(avg_freq, avg_amp, avg_roomSize)
        else:
            self.audio_gen.stop_audio()

    def on_close(self):
        self.audio_gen.stop_audio()
        # Cancel any pending 'after' calls
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.root.destroy()

    def visualizeReverb(self, roomSize):
        self.ax.clear()  # Clear the previous plot

        r_val = roomSize / 2
        r = [-r_val, r_val]
        vertices = np.array(list(product(r, r, r)))

        # Define the 6 faces of the cube (important for Poly3DCollection)
        faces = [
            [vertices[0], vertices[1], vertices[3], vertices[2]],  # Front face
            [vertices[4], vertices[5], vertices[7], vertices[6]],  # Back face
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # Bottom face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # Top face
            [vertices[0], vertices[2], vertices[6], vertices[4]],  # Left face
            [vertices[1], vertices[3], vertices[7], vertices[5]]  # Right face
        ]

        # Plot the cube as a collection of polygons for better visualization
        self.ax.add_collection3d(Poly3DCollection(faces, facecolors=None, linewidths=1, edgecolors='r', alpha=0.0))

        elev_Val = 50 * time.time() # vertical rotation with time
        roll_Val = 20 * time.time() # horizontal rotation with time

        # Update axis limits dynamically -- might hide
        self.ax.set_axis_off()
        self.ax.set_facecolor((0.0,0.0,0.0,0.0))
        self.ax.set_xlim([-r_val, r_val])
        self.ax.set_ylim([-r_val, r_val])
        self.ax.set_zlim([-r_val, r_val])
        self.ax.view_init(elev=elev_Val, azim=45, roll=roll_Val ) #elev:elevation, azim:azimuth to get desired perspective for the cube
        self.ax.set_title(f"Reverb Room Size: {roomSize:.2f}")

        # Draw the updated plot
        self.fig.canvas.draw()
        buf = self.fig.canvas.tostring_argb() #getrgba data from the figure's canvas
        pil_image = Image.frombuffer('RGBA', self.fig.canvas.get_width_height(), buf, 'raw', 'ARGB', 0, 1) #convert to pil image (mode rgba)
        self.cube_image = pil_image.convert('RGBA')


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()