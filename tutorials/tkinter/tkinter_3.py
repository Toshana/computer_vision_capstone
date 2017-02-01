# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 14:21:39 2016

@author: Toshana
"""

from Tkinter import *
import cv2

cam = cv2.VideoCapture(0)
s, im = cam.read() # captures image
cv2.imshow("Test Picture", im) # displays captured image
cv2.imwrite("test.png",im) # writes image test.png to disk