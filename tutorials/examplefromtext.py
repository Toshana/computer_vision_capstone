# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:45:17 2017

@author: centraltendency
"""

# Another way?

import cv2
import os
from sklearn.cross_validation import train_test_split # this will change to sklearn.model_selection in 0.20
from sklearn import preprocessing
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_data(location):
    files = []
    labels = []    
    for f in os.listdir(location):
        new_location = location + '/' + f
        for pic in os.listdir(new_location):
            image = new_location + '/' + pic
            files.append(cv2.imread(image, 0))
            labels.append(f)
    return files, labels
    
def split_data(x, y):
    le = preprocessing.LabelEncoder()
    le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 12)
    return X_train, X_test, y_train, y_test    

file_loc = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories"
files, labels = load_data(file_loc)

X_train, X_test, y_train, y_test = split_data(files, labels)

detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

flann_params = dict(algorithm = 1, trees = 5)
flann = cv2.FlannBasedMatcher(flann_params, {})

bow_kmeans_trainer = cv2.BOWKMeansTrainer(101)
extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)

def extract_sift(image):
    return extract.compute(image, detect.detect(image))[1]
    
for x in X_train:
    bow_kmeans_trainer.add(extract_sift(x))

vocabulary = bow_kmeans_trainer.cluster()
extract_bow.setVocabulary(vocabulary)

def bow_features(image):
    return extract_bow.compute(image, detect.detect(image))
    
train_data = []
train_labels = []
