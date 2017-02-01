# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:23:30 2016

@author: Toshana

This is taken from the book "Learning OpenCV3 Computer Vision with Python Second Edition"
pg 122
"""

import cv2
import numpy as np
import os
import sys

# SIFT (generate descriptors)

#def create_descriptors(folder):
#    files = []
#    for (dirpath, dirnames, filenames) in walk(folder):
#        files.extend(filenames)
#        for f in files:
#            save_descriptor(folder, f, cv2.ORB())
            
def create_descriptors(file_location):
    files = []
    for name in os.listdir(file_location):
        new_location = file_location + '\\' + name
        for pic in os.listdir(new_location):
            image = new_location +'\\' + pic
            files.extend(image)
            for f in files:
                save_descriptor(file_location, f, cv2.ORB)
            
def save_descriptor(folder, image_path, feature_detector):
    img = cv2.imread(join(folder, image_path), 0)
    keypoints, descriptors = feature_detector.detectAndCompute(img, None)
    descriptor_file = image_path.replace("jpg", "npy")
    np.save(join(folder, descriptor_file), descriptors)
    dir = sys.argv[1]
    create_descriptors(dir)