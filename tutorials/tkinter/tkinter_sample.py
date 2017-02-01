# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 13:59:12 2016

@author: Toshana
"""
from Tkinter import *

root = Tk()

w = Label(root, text="Hello, world!") # create a label widget as a child to the root window
w.pack() # tells it to resize itself to the given text and make itself visible

root.mainloop() # makes the application appear
