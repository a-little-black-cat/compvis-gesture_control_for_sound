import self
from PIL import Image as PILImage, ImageTk
from tkinter import *
import os
import pytesseract
import mediapipe as mp
from mediapipe.python import *
import cv2 #pip install opencv-contrib-python got rid of the 'cv2 not found' error

##camera
cap = cv2.VideoCapture(0)  # default camera but will add a camera select option.
width, height = 640, 480  # Width of camera, Height of camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#initializing mediapipe
def __init__(self,mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
    self.__mode__ = mode
    self.__maxHands__ = maxHands
    self.__detectionCon__ = detectionCon
    self.__trackCon__ = trackCon
    self.handsMp = mp.solutions.hands
    self.hands = self.handsMP.Hands()
    self.mpDraw=mp.solutions.drawing_utils
    self.tipIds = [4, 8, 12, 16, 20]

#if not cap.isOpened():
#    raise IOError("Cannot open camera")

window = Tk()
window.bind('<Escape>', lambda e: window.quit())
def findFingers(self, frame, draw=True):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)
    if self.results.multi_hand_landmarks:
        for handLms in self.results.multi_hand_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, handLms,self.handsMp.HAND_CONNECTIONS)
            return frame

def findPosition(self, frame, handNo=0,  draw=True):
    xList=[]
    yList=[]
    bbox = []
    self.lmsList=[]
    if not self.results.multi_hand_landmarks:
        myHand = self.results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            h,w,c = frame.shape # c -> center
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            self.lmsList.append([id,cx,cy])
            if draw:
                cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)

        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = xmin, ymin, xmax, ymax
        if draw:
            cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),(0,25,0),2)
        return self.lmsList, bbox

def open_camera():
    ret, frame = cap.read()
    opencv_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA)
    captured_image = PILImage.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    label_widget.after(10, open_camera)
    button1.pack_forget()
    button2.pack()



def close_camera():
    cap.release()
    button1.pack()
    button2.pack_forget()



button1 = Button(window, text="Open Camera", command = open_camera)
button1.pack()
button2 = Button(window, text="Close Camera", command = close_camera)
window.title("Camera Text Reader")
window.geometry("1000x1000")
label_widget = Label(window)
label_widget.pack()

label = Label(window, text="Welcome to Camera Text Reader")
label.pack()

window.mainloop()