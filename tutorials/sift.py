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
import pickle

class trainData():
    
    def __init__(self):
        self.files = []
        self.labels = []
        self.vocabulary_size = 102        
        self.BOW_train = cv2.BOWKMeansTrainer(self.vocabulary_size)
        self.BOW_test = cv2.BOWKMeansTrainer(self.vocabulary_size)
        

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
                
    def create_descriptors(self, x, BOW):
        sift = cv2.xfeatures2d.SIFT_create()
        for f in x:
            kp, des = sift.detectAndCompute(f, None)
            BOW.add(des)
            
    # Cluster the descriptors to create a codebook
    
    def create_codebook(self, BOW_trainer, file_name):
        vocabulary = BOW_trainer.cluster()
        fileObject = open(file_name, 'wb')
        pickle.dump(vocabulary, fileObject)
        fileObject.close()
        
    def get_vocabulary(self, file_location):
        fileObject = open(file_location, "r")
        self.vocabulary = pickle.load(fileObject)
        
    def create_imgDescriptor(BOW, vocabulary):
        sift2 = cv2.DescriptorExtractor_create("SIFT")
        bf = cv2.BFMatcher(cv2.NORM_L2)
        extractor = cv2.BOWImgDescriptorExtractor(sift2, bf)
        extractor.setVocabulary(vocabulary)

        
            
            

    
    

            
        
    
