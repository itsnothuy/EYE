from __future__ import print_function
import cv2 as cv
import argparse
import time

def detectAndDisplay(frame, face_cascade, eyes_cascade):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    
    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        
        faceROI = frame_gray[y:y+h, x:x+w]
        
        # -- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
            
    cv.imshow('Capture - Face detection', frame)

# Load the cascades
face_cascade = cv.CascadeClassifier(cv.samples.findFile('haarcascade_frontalface_alt.xml'))
eyes_cascade = cv.CascadeClassifier(cv.samples.findFile('haarcascade_eye_tree_eyeglasses.xml'))

if not face_cascade.load(cv.samples.findFile('haarcascade_frontalface_alt.xml')):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile('haarcascade_eye_tree_eyeglasses.xml')):
    print('--(!)Error loading eyes cascade')
    exit(0)

camera_device = 0

# Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    
    detectAndDisplay(frame, face_cascade, eyes_cascade)
    
    key = cv.waitKey(10)
    if key == 27:  # ESC key to exit
        break
    elif key == ord("s") or key == ord("S"):
        t = time.localtime()
        filename = f'image_{t.tm_year}_{t.tm_mon:02d}_{t.tm_mday:02d}_{t.tm_hour:02d}_{t.tm_min:02d}_{t.tm_sec:02d}.jpg'
        cv.imwrite(filename, frame)
        print(f'Saved {filename}')
