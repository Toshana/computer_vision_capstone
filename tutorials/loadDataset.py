# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 14:40:48 2017

@author: Toshana
"""

import cv2
import numpy as np
import os
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer


#file_location = "C:\Users\Toshana\Documents\Udacity\computer_vision\ObjectCategories\accordion"
#os.path.exists(file_location)
#
#files = []
#for name in os.listdir('.'):
#    files.append(name)
#    
#x = [np.array(Image.open(fname)) for fname in files]

# Another way

#image_eg = "C:/Users/Toshana/Documents/Udacity/computer_vision/ObjectCategories/airplanes/image_0001.jpg"
file_location = "C:\Users\Toshana\Documents\Udacity\computer_vision\ObjectCategories"
data = []
target = []
dataset = {}
# dataset = dataset.fromkeys(["data", "target"], [])
for name in os.listdir(file_location):
    new_location = file_location + '\\' + name
    for pic in os.listdir(new_location):
        image = new_location +'\\' + pic
        x = [np.array(Image.open(image))]
        data.append(x)
        target.append(name)
dataset["data"] = np.array(data)
dataset["target"] = np.array(target)

def get_descriptors(dataset):
    orb = cv2.ORB()
    descriptor_matrix = np.zeros((1, 128))
    for item in dataset:
        keypoints, descriptors = orb.detectAndCompute(item, None)
        descriptor_matrix = np.concatenate((descriptor_matrix, descriptors), axis = 0)
    descriptor_matrix = descriptor_matrix[1:, :]



# from Handwriting example

if __name__ == '__main__':
#    data = dataset
    X, y = dataset["data"], dataset["target"]
#    MultiLabelBinarizer().fit_transform(dataset)
#    X = X/255.0*2 - 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    pipeline = Pipeline([
        ('clf', SVC(kernel='rbf', gamma = 0.01, C = 100))
    ])
    print X_train.shape
    parameters = {
        'clf__gamma': (0.01, 0.03, 0.1, 0.3, 1),
        'clf__C': (0.1, 0.3, 1, 3, 10, 30),
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs = -1, verbose = 1, scoring = 'accuracy')
    grid_search.fit(X_train[:10000], y_train[:10000])
    print 'Best score: %0.3f' % grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])
        predictions = grid_search.predict(X_test)
        print classification_report(y_test, predictions)


