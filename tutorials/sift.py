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
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

class trainData():
    
    def __init__(self):
        self.files = []
        self.labels = []
        self.vocabulary_size = 102        
        self.BOW = cv2.BOWKMeansTrainer(self.vocabulary_size)
        

    # Load dataset
    def load_data(self, location):
        for f in os.listdir(location):
            new_location = location + '/' + f
            for pic in os.listdir(new_location):
                image = new_location + '/' + pic
                self.files.append(cv2.imread(image, 0))
                self.labels.append(f)
    
    # Create Training and Testing Data
    
    def split_data(self, x, y):
        le = preprocessing.LabelEncoder()
        le.fit_transform(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.1, random_state = 7)
                
    # ORB (generate descriptors)
                
    def create_descriptors(self, x1, x2):
        '''
        
        '''
        self.orb = cv2.ORB()
        count = 0
        for f in x1:
            count += 1
            kp, des = self.orb.detectAndCompute(f, None)
            self.BOW.add(des)
            
            
            
    img = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories/accordion/image_0001.jpg"
    file_loc = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories"
    des_loc = "/home/centraltendency/Udacity/computer_vision_capstone/descriptors/descriptor"
    create_descriptors(file_loc, des_loc)
    
    
    # Cluster the descriptors to create a codebook
    
    def create_codebook(BOW_trainer):
        BOW.cluster()
        
            
        
    
