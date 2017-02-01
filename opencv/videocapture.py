# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 12:12:33 2016

@author: Toshana
"""

# Video capture with open cv

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

cap2 = cv2.VideoCapture("vidtest.avi")

while(cap2.isOpened()):
    ret, frame = cap2.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap2.release()
cv2.destroyAllWindows