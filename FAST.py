# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 22:52:47 2017

@author: centraltendency
"""
import sys
sys.path.remove('/usr/lib/python2.7/dist-packages')


import cv2
import os
from sklearn.model_selection import train_test_split # this will change to sklearn.model_selection in 0.20
from sklearn import preprocessing
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means


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
    new_y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(x, new_y, test_size = 0.3, random_state = 12)
    return X_train, X_test, y_train, y_test    

def get_descriptors(X, n):
    orb = cv2.ORB_create()
    bow = cv2.BOWKMeansTrainer(n)
    for x in X:
        kp = orb.detect(x, None)        
        kp, des = orb.compute(x, kp)
        bow.add(des)
    return bow
    
def train_data(X, bow):
    traindata = []
    for x in X:
        traindata.extend(bow.compute(x, detect.detect(x)))
    return traindata

    
file_loc = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories"
X, Y = load_data(file_loc)
xtrain, xtest, ytrain, ytest = split_data(X, Y)
bow = get_descriptors(xtrain, 101)
clusters = bow.cluster()
detect = cv2.FastFeatureDetector_create()
extract = cv2.FastFeatureDetector_create()
flann_params = dict(algorithm = 1, trees = 4)
flann = cv2.FlannBasedMatcher(flann_params, {})
extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)
extract_bow.setVocabulary(clusters)

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

clf = SVC()

X_train = np.array(train_data(xtrain, extract_bow))
y_train = ytrain
X_test = np.array(train_data(xtest, extract_bow))
y_test = ytest

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits = 5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
