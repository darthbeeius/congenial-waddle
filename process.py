import numpy as np
import cv2
cap = cv2.VideoCapture("E:\AT3_6.mp4")
ret, frame = cap.read()
r, h, c, w = 150, 100, 10, 125
track_window = (c, r, w, h)
out = cv2.VideoWriter('output3.avi', -1, 20.0, (125,100))
roi = frame[r:r + h, c:c + w]
while (1):  
    ret, frame = cap.read()
    if ret:
        x, y, w, h = track_window
        dst= frame[y:y+h,x:x+w]
        out.write(dst)
        cv2.imshow('dst',dst)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
    else:
        break