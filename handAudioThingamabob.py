import cv2
import mediapipe as mp
import threading
import numpy as np
import sounddevice as sd
import tkinter as tk
from PIL import Image, ImageTk
from pysndfx import AudioEffectsChain
import math

fx = (
    AudioEffectsChain()
    .highshelf()
    .reverb()
)

class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.drawing_utils = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results

class AudioGeneration:
    def __init__(self):
        self.fs = 16000  # Sample rate
        self.freq = 440.0
        self.amplitude = 0.2
        self.running = False
        self.lock = threading.Lock()
        self.phase = 0.0  # Phase in radians
        self.roomSize = 0.3


    def start_audio(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.audioGen, daemon=True).start()

    def stop_audio(self):
        with self.lock:
            self.running = False

    def audioGen(self):
        def callback(outdata, frames, time, status):
            with self.lock:
                if not self.running:
                    raise sd.CallbackStop()

                t = (np.arange(frames) + self.phase_offset) / self.fs
                samples = self.amplitude * np.sin(2 * np.pi * self.freq * t).astype(np.float32)

                outdata[:, 0] = samples

                self.phase_offset += frames
                self.phase_offset %= self.fs
        self.phase_offset = 0
        with sd.OutputStream(channels=1, callback=callback, samplerate=self.fs, dtype='float32', blocksize=1024):
            while True:
                with self.lock:
                    if not self.running:
                        break
                sd.sleep(100)


    def set_frequency(self, freq):
        with self.lock:
            self.freq = freq

    def set_amplitude(self,amplitude):
        with self.lock:
            self.amplitude=amplitude

    def set_room_size(self, roomSize):
        with self.roomSize:
            self.roomSize = roomSize

    def set_parameters(self, freq, amplitude, roomSize):
        with self.lock:
            alpha = 0.2 # smoothing factor ,, how much influence the new value has compared to the old value (self.freq, self.ampl
            self.freq = self.freq * (1-alpha) + freq * alpha
            self.amplitude = self.amplitude * (1-alpha) + amplitude * alpha
            self.roomSize = self.roomSize * (1-alpha) + roomSize * alpha
            # exponential smoothing: low-pass filtering.





class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracker with Audio")

        self.cap = cv2.VideoCapture(0)
        self.hand_tracker = HandTracker()
        self.audio_gen = AudioGeneration()

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.frame_count = 0
        self.freq_buffer = []
        self.amp_buffer = []
        self.roomSize_buffer = []
        self.smoothing_window = 5
        self.update_interval = 5  # update audio every N frames

        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            results = self.hand_tracker.detect_hands(frame)

            if results.multi_hand_landmarks:
                self.audio_gen.start_audio()
                for hand_landmarks in results.multi_hand_landmarks:
                    self.hand_tracker.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )

                    # Palm position (landmark 0)
                    palm_pos = hand_landmarks.landmark[0]

                    # Map palm_x to frequency
                    palmpos_x = 1.0 - palm_pos.x
                    freq = 220 + (880 - 220) * palmpos_x

                    # Map palm_y to amplitude
                    palmpos_y = 1.0 - palm_pos.y
                    amplitude = 0.1 + 0.4 * palmpos_y

                    # Store recent values for smoothing
                    self.freq_buffer.append(freq)
                    self.amp_buffer.append(amplitude)
                    self.roomSize_buffer.append(room_size)

                    if len(self.freq_buffer) > self.smoothing_window:
                        self.freq_buffer.pop(0)
                        self.amp_buffer.pop(0)
                        self.roomSize_buffer.pop(0)

                    self.frame_count += 1
                    if self.frame_count % self.update_interval == 0:
                        avg_freq = sum(self.freq_buffer) / len(self.freq_buffer)
                        avg_amp = sum(self.amp_buffer) / len(self.amp_buffer)
                        avg_roomSize = sum(self.roomSize_buffer) / len(self.roomSize_buffer)
                        self.audio_gen.set_parameters(avg_freq, avg_amp,avg_roomSize)

                    # Mapping distance between thumb and index for reverb -- 4: thumb tip | 8: index tip
                    thumbTip_pos = hand_landmarks.landmark[4]
                    indexTip_pos = hand_landmarks.landmark[8]

                    ## one issue i might face is that if the hand is close to the screen the difference will appear larger, as whereas if it is farther away the difference will appear smaller.
                    ## however, this can actually be utilised for a larger range?
                    ## thumbtip_pos x - indextip_pos x & thumbtip_pos y - indextip_pos y ,, then grab hypotenuse?
                    thumbTip_posX = 1.0 - thumbTip_pos.x
                    thumbTip_posY = 1.0 - thumbTip_pos.y

                    indexTip_posX = 1.0 - indexTip_pos.x
                    indexTip_posY = 1.0 - indexTip_pos.y

                    thumbTip_posXY = [thumbTip_posX, thumbTip_posY]

                    indexTip_posXY = [indexTip_posX, indexTip_posY]

                    tipDistance_reverb = math.dist(thumbTip_posXY,indexTip_posXY)
                    print(tipDistance_reverb)
                    # using the Eyring Equation

                    volume_reverb = pow(tipDistance_reverb, 3) #volume of a cubic environment
                    scaled_volume = min(max(volume_reverb * 100, 0), 1.0)  # Clamp to [0, 1]

                    room_size = min(max(volume_reverb * 5, 0.1), 1.0)
                    Surface = pow(tipDistance_reverb,2) * 6 ##for a cubic environment
                    absorption = 0.35 ## will modify so that it is customizable


                    reverbTime = -0.161*scaled_volume/(Surface * np.log(1-absorption))

                    print(volume_reverb)

            else:
                self.audio_gen.stop_audio()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.imgtk = img

        self.root.after(10, self.update_video)

    def on_close(self):
        self.audio_gen.stop_audio()
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

## To Do:
# Add a slider to adjust starting frequence
# buttons to adjust soundwave
# reverb control thumb and index tips.
# can feed outstream from sd into the system again to apply effects?
#

