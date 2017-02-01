# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 14:11:34 2016

@author: Toshana
"""

import time
from SimpleCV import Camera

cam = Camera()
time.sleep(0.1)  # If you don't wait, the image will be dark
img = cam.getImage()
img.save("simplecv.png")
del cam #does not work

import cv2

cam = cv2.VideoCapture(0)
s, im = cam.read()
cv2.imshow("testpic", im)
cv2.imwrite("test.png", im)
del cam