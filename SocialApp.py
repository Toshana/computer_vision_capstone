# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 11:43:48 2016

@author: Toshana
"""

from Tkinter import *
import cv2
import numpy as np
import time
from SimpleCV import Camera

class ImageApp:
    
    def __init__(self, master):
        
        frame = Frame(master)
        frame.pack()
        
        self.buttonA = Button(frame, text = "Upload Image", command = self.upload_image)
        self.buttonA.pack(side = LEFT)
        
        self.buttonB = Button(frame, text = "Take Picture", command = self.take_picture)
        self.buttonB.pack(side = LEFT)
        
        self.buttonC = Button(frame, text = "Describe Image", command = self.describe_image)
        self.buttonC.pack(side = LEFT)
        
    def upload_image(self):
        print "This button uploads a saved image!"
        
    def take_picture(self):
        cam = cv2.VideoCapture(0)
        s, im = cam.read()
        cv2.imshow("testpic", im)
        cv2.imwrite("test.png", im)
        del cam
        print "This button displays image from webcam and saves on click!"
        
#    def show_webcam(mirror = False):
#        cam = cv2.VideoCapture(0)
#    
#        while True:
#            ret_val, img = cam.read()
#            if mirror:
#                img = cv2.flip(img, 1)
#                cv2.imshow('my_webcam', img)
#            if cv2.waitKey(1) == 27:
#                break
#            cv2.destroyAllWindows()
        
    def describe_image(self):
        print "This button shows a random image that user can describe!"
        


root = Tk()
app = ImageApp(root)
root.mainloop()
        