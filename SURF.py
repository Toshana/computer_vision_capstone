# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:11:42 2017

@author: centraltendency
"""

# SURF

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

# Check keypoints on sample image

img = X[0]
surf = cv2.xfeatures2d.SURF_create(400)
surf_300 = cv2.xfeatures2d.SURF_create(1000)
kp, des = surf_300.detectAndCompute(img, None)
print len(kp)
img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), 4)
plt.imshow(img2),plt.show()

# Detect keypoints using SURF

def detect_surf(X, n_clusters):
    surf = cv2.xfeatures2d.SURF_create(400)
    bow = cv2.BOWKMeansTrainer(n_clusters)
    for f in X:
        kp, des = surf.detectAndCompute(f, None)
        bow.add(des)
    return bow

def detect_surf_1000(X, n_clusters):
    surf = cv2.xfeatures2d.SURF_create(1000)
    bow = cv2.BOWKMeansTrainer(n_clusters)
    for f in X:
        kp, des = surf.detectAndCompute(f, None)
        bow.add(des)
    return bow
    
# cluster keypoints
    
def cluster_kp(bow, n):
    kmeans = k_means(bow, n_clusters = n, random_state = 8)
    return kmeans
    
def cluster_surf(bow):
    return bow.cluster()
    
# initialize bow extractor

detect = cv2.xfeatures2d.SURF_create()
extract = cv2.xfeatures2d.SURF_create()
flann_params = dict(algorithm = 1, trees = 4)
flann = cv2.FlannBasedMatcher(flann_params, {})
# extract_bow = cv2.BOWImageDescriptorExtractor(extract, flann) THIS IS THE EXTRACTOR

# pre-process data for training

def train_data(X):
    traindata = []
    for x in X:
        traindata.extend(extract_bow.compute(x, detect.detect(x)))
    return traindata

def train_data_1000(X):
    traindata = []
    for x in X:
        traindata.extend(extract_bow1000.compute(x, detect.detect(x)))
    return traindata    
# helper functions for saving clusters
    
def save_pickle(item, filename):
    fileObject = open(filename, 'wb')
    pickle.dump(item, fileObject)
    fileObject.close()
    
def load_pickle(filelocation):
    fileObject = open(filelocation, "r")
    items = pickle.load(fileObject)
    return items
    
file_loc = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories"
files, labels = load_data(file_loc)
X_train, X_test, y_train, y_test = split_data(files, labels)

surfbow = detect_surf(X_train, 1000)
surf_1000 = detect_surf_1000(X_train, 101)

# clusters = cluster_kp(des_array, 101) # did not worrk
#cluster_pickle = "/home/centraltendency/Udacity/computer_vision_capstone/training/kmeansCluster"
#save_pickle(clusters, cluster_pickle)

surf_clusters = cluster_surf(surfbow)
extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)
extract_bow.setVocabulary(surf_clusters)

surf1000_clusters = cluster_surf(surf_1000)
extract_bow1000 =  cv2.BOWImgDescriptorExtractor(extract, flann)
extract_bow1000.setVocabulary(surf1000_clusters)

# Create SVM

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

clf = SVC()

# Train data

traindata = train_data(X_train)
trainlabels = y_train
testdata = train_data(X_test)
testlabels = y_test

clf.fit(np.array(traindata), np.array(trainlabels))

# make predictions

predictions = clf.predict(np.array(testdata))

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits = 5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(np.array(traindata), np.array(trainlabels))
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

traindata1000 = train_data_1000(X_train)
trainlabels = y_train
testdata1000 = train_data_1000(X_test)
testlabels = y_test


C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits = 5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(np.array(traindata1000), np.array(trainlabels))
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
