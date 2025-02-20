import os
import cv2

 
capture = cv2.VideoCapture('Match_1864_1_0_subclip.mp4')
 
frameNr = 0
 
while (True):
 
    success, frame = capture.read()
 
    if success:
        cv2.imwrite(f'outputs/frame{frameNr}.jpg', frame)
 
    else:
        break
 
    frameNr = frameNr+1
 
capture.release()