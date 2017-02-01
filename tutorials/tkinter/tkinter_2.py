# -*- coding: utf-8 -*-
"""
Created on Sat Dec 03 09:56:00 2016

@author: Toshana
"""

# our second tkinter program

from Tkinter import *

class App:
    
    def __init__(self, master):
        
        frame = Frame(master)
        frame.pack()
        
        self.button = Button(frame, text="QUIT", fg="red", command=frame.quit)
        self.button.pack(side=LEFT)
        
        self.hi_there = Button(frame, text="HELLO", command=self.say_hi)
        self.hi_there.pack(side=LEFT)
        
    def say_hi(self):
        print "hi there everyone!"

root = Tk()
app = App(root)
root.mainloop()
#root.destroy()

# this example does not work for some reason

class App:

    def __init__(self, master):

        frame = Frame(master)
        frame.pack()

        self.button = Button(
            frame, text="QUIT", fg="red", command=frame.quit
            )
        self.button.pack(side=LEFT)

        self.hi_there = Button(frame, text="Hello", command=self.say_hi)
        self.hi_there.pack(side=LEFT)

    def say_hi(self):
        print "hi there, everyone!"

root = Tk()

app = App(root)

root.mainloop()
root.destroy() # optional; see description below

# this one does work. what exactly is "button"?

import numpy as np
import cv2
 
cap = cv2.VideoCapture(0)
 
while(True):
 # Capture frame-by-frame
    ret, frame = cap.read()
 
 # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
 # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
 # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
