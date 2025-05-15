import cv2
import time
import mediapipe as mp
import pytesseract
from tkinter import *
from PIL import Image as PILImage, ImageTk

class DetectText:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None
        self.lmsList = []

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmsList = []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return self.lmsList, bbox

    def findFingerUp(self):
        fingers = []
        if not self.lmsList:
            return fingers
        # Thumb
        if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers
        for id in range(1, 5):
            if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

# Global variables
cap = None
detector = DetectText()

def open_camera():
    global cap
    cap = cv2.VideoCapture(0)
    update_frame()
    button1.pack_forget()
    button2.pack()

def close_camera():
    global cap
    if cap:
        cap.release()
    label_widget.config(image='')
    button1.pack()
    button2.pack_forget()

def update_frame():
    global cap
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = detector.findFingers(frame)
            lmsList, _ = detector.findPosition(frame)
            opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            captured_image = PILImage.fromarray(opencv_image)
            photo_image = ImageTk.PhotoImage(image=captured_image)
            label_widget.photo_image = photo_image
            label_widget.configure(image=photo_image)
        label_widget.after(10, update_frame)

# GUI setup
window = Tk()
window.title("opencv-mediapipe CG")
window.geometry("1000x800")
window.bind('<Escape>', lambda e: window.quit())

label = Label(window, text="CG", font=("Helvetica", 16))
label.pack(pady=10)

label_widget = Label(window)
label_widget.pack()

button1 = Button(window, text="Open Camera", command=open_camera)
button1.pack(pady=5)

button2 = Button(window, text="Close Camera", command=close_camera)
button2.pack_forget()

window.mainloop()
