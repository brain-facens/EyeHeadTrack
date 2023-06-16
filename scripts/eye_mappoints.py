import cv2
import dlib
import numpy as np
from contextlib import redirect_stdout
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import pandas as pd
import os 
import sys
sys.path.append('../')
from utils.eye import EyeTrack
from utils.plot_heatmap import plot_heatmap
import datetime

try:
    os.remove("../extras/coordinates.csv")
except:
    pass

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 117, 255, nothing)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        start_time = datetime.datetime.now().strftime("%H:%M:%S")
        shape = predictor(gray, rect)
        shape = EyeTrack.shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = EyeTrack.eye_on_mask(mask, left, shape)
        mask = EyeTrack.eye_on_mask(mask, right, shape)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        EyeTrack.contouring(thresh[:, 0:mid], mid, img)
        EyeTrack.contouring(thresh[:, mid:], mid, img, True)
        mid_x = shape[42][0]
        mid_y = shape[39][0]
        print(mid_x, mid_y, start_time)
        with open("../extras/coordinates.csv", "a+") as f:
            with redirect_stdout(f):
                print(str(mid_x) + "," + str(mid_y) + "," + start_time)
                
    # show the image with the face detections + facial landmarks
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
df = pd.read_csv("../extras/coordinates.csv")
plot_heatmap(df.iloc[:, 0], df.iloc[:, 1])