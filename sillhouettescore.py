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
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


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

file_loc = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories"
files, labels = load_data(file_loc)
X_train, X_test, y_train, y_test = split_data(files, labels)

x = X_train

descriptors = []
for f in x:
    surf = cv2.xfeatures2d.SURF_create()
    kp, des = surf.detectAndCompute(f, None)
    descriptors.extend(des)
    
#descriptors_list = []
#for f in x:
#    surf = cv2.xfeatures2d.SURF_create()
#    kp, des = surf.detectAndCompute(f, None)
#    descriptors_list.append(des)
#
#X = np.array(descriptors)
#K = range(1, 1000, 100)
#meandistortions = []
#for k in K:
#    kmeans = KMeans(n_clusters = k)
#    kmeans.fit(X)
#    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, "euclidean"), axis = 1)) / X[0])

X = np.array(descriptors)
distortions_list = []
for i in range (1, 1001, 100):
    km = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, random_state = 0)
    km.fit(X)
    distortions_list.append(km.inertia_)

plt.plot(range(1, 11), distortions_list, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

#plt.plot(K, meandistortions, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Average Distortion')
#plt.title('Selecting K with the Elbow Method')
#plt.show()