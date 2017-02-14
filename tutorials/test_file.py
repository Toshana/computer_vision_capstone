# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:26:28 2017

@author: centraltendency
"""

import cv2
import os
from sklearn.cross_validation import train_test_split # this will change to sklearn.model_selection in 0.20
from sklearn import preprocessing
import pickle
import numpy as np
import matplotlib.pyplot as plt

files = []
labels = []
vocabulary_size = 101
BOW_train = cv2.BOWKMeansTrainer(vocabulary_size)
BOW_test = cv2.BOWKMeansTrainer(vocabulary_size)
X_train = []
X_test = []
y_train = []
y_test = []

def load_data(location):
    for f in os.listdir(location):
        new_location = location + '/' + f
        for pic in os.listdir(new_location):
            image = new_location + '/' + pic
            files.append(cv2.imread(image, 0))
            labels.append(f)
            
def split_data(x, y):
    le = preprocessing.LabelEncoder()
    le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 7)
    
def create_descriptors(x):
    sift = cv2.xfeatures2d.SIFT_create()
    for f in x:
        kp, des = sift.detectAndCompute(f, None)
        BOW_train.add(des)
        
def create_codebook(BOW_trainer, file_name):
    vocabulary = BOW_train.cluster()
    fileObject = open(file_name, 'wb')
    pickle.dump(vocabulary, fileObject)
    fileObject.close()
    
def get_vocabulary(file_location):
    fileObject = open(file_location, "r")
    vocabulary = pickle.load(fileObject)
    
train_data = []
train_labels = []
def create_imgDescriptor(BOW, x):
    sift = cv2.xfeatures2d.SIFT_create()    
    sift2 = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    extractor = cv2.BOWImgDescriptorExtractor(sift2, bf)
    extractor.setVocabulary(vocabulary)
    count = 0
    for img in x:        
        count += 1
        print "Working on item number {}".format(count)
        kp = sift.detect(img, None)
        des = extractor.compute(img, kp)
        train_data.append(des)
        
        
        

img = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories/accordion/image_0001.jpg"
file_loc = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories"
des_loc = "/home/centraltendency/Udacity/computer_vision_capstone/descriptors/descriptor"
pickle_loc = "/home/centraltendency/Udacity/computer_vision_capstone/training/BOW_pickle"

load_data(file_loc)

le = preprocessing.LabelEncoder()
le.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size = 0.1, random_state = 7)

create_descriptors(X_train)
create_descriptors(X_test)

fileObject = open(pickle_loc, "r")
vocabulary = pickle.load(fileObject)

create_imgDescriptor(vocabulary, X_train)

# Train SVM
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report


x = np.asarray(train_data)
y = y_train

nsamples, nx, ny = x.shape
d_dataset = x.reshape((nsamples, nx*ny))
x = d_dataset

clf = SVC()
clf.fit(x, y)

pipeline = Pipeline([
        ('clf', SVC(kernel='rbf', gamma = 0.01, C = 100))
    ])
print x.shape
parameters = {
    'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
    'clf__C': (0.1, 0.3, 1, 3, 10, 30),
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs = -1, verbose = 1, scoring = 'accuracy')
grid_search.fit(x[:10000], y[:10000])
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])
    predictions = grid_search.predict(X_test)
    print classification_report(y_test, predictions)
