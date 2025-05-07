from PIL import Image as PILImage, ImageTk
from tkinter import *
import pytesseract
import cv2 #pip install opencv-contrib-python got rid of the 'cv2 not found' error

##camera
cap = cv2.VideoCapture(0) #default camera but will add a camera select option.
width, height = 640, 480  # Width of camera, Height of camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)



if not cap.isOpened():
    raise IOError("Cannot open camera")

window = Tk()
window.bind('<Escape>', lambda e: window.quit())

def open_camera():
    _, frame = cap.read()
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