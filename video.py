import cv2 as cv
import time

camera_device = 0

cap = cv.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No capture fram -- Break!')
        break
    
    cv.imshow('video', frame)
    
    key = cv.waitKey(10)
    
    if key == 27:
        break
    if key == ord("s") or key == ord("S"):
        t = time.localtime()
        filename = 'image_%04d_%02d_%02d_%02d_%02d_%02d_%02d.jpg'
        (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
        cv.imwrite(filename, frame)
        
