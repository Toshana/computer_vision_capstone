# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:26:28 2017

@author: centraltendency
"""
import sys
sys.path.remove('/usr/lib/python2.7/dist-packages')


import cv2
import os
from sklearn.cross_validation import train_test_split # this will change to sklearn.model_selection in 0.20
from sklearn import preprocessing
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_data(location):
    """
    Takes the location of a directory with named folders of images.
    Returns images and labels.     
    """
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
    """
    Takes images and labels and splits the data into training and testing sets.    
    """
    le = preprocessing.LabelEncoder()
    new_y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(x, new_y, test_size = 0.3, random_state = 12)
    return X_train, X_test, y_train, y_test    
    
def create_descriptors(x, vocabulary_size):
    """
    Initialize K-Means Bag of Words trainer from OpenCV and adds SIFT descriptors for clustering.    
    """
    BOW_train = cv2.BOWKMeansTrainer(vocabulary_size)
    sift = cv2.xfeatures2d.SIFT_create()
    for f in x:
        kp, des = sift.detectAndCompute(f, None)
        BOW_train.add(des)
    return BOW_train
        
def create_codebook(BOW_train, file_name):
    """
    Clusters Bag of Words object.
    Saves as pickle to location "file_name".
    """
    vocabulary = BOW_train.cluster()
    fileObject = open(file_name, 'wb')
    pickle.dump(vocabulary, fileObject)
    fileObject.close()
    return vocabulary
    
def get_vocabulary(file_location):
    """
    Takes location of saved pickle and returns python object.    
    """
    fileObject = open(file_location, "r")
    vocabulary = pickle.load(fileObject)
    return vocabulary
    
def create_imgDescriptor(BOW, x):
    """
    Initializes SIFT detector and extractor along with the Brute Force matching algorithm.
    Takes Bag of Words vocabulary and list of training images, and returns the SIFT descriptors for that vocabulary.
    """
    detect = cv2.xfeatures2d.SIFT_create()    
    extract = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    extractor = cv2.BOWImgDescriptorExtractor(extract, bf)
    extractor.setVocabulary(BOW)
    train_data = []
    for img in x:        
        kp = detect.detect(img, None)
        des = extractor.compute(img, kp)
        train_data.extend(des)
    return train_data        
 
def create_flann_imgDescriptor(BOW, x):
    """
    Initializes SIFT detector and extractor along with the FLANN matching algorithm.
    Takes Bag of Words vocabulary and list of training images, and returns the SIFT descriptors for that vocabulary.        
    """
    detect = cv2.xfeatures2d.SIFT_create()    
    extract = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
    fm = cv2.FlannBasedMatcher(flann_params, {})
    extractor = cv2.BOWImgDescriptorExtractor(extract, fm)
    extractor.setVocabulary(BOW)
    train_data = []
    for img in x:        
        kp = detect.detect(img, None)
        des = extractor.compute(img, kp)
        train_data.extend(des)
    return train_data        

    
def bow_features(im, vocab):
    """
    Taken from "OpenCV Computer Vision with Python" by Joseph Howse.  
    """
    i = cv2.imread(im)
    extract = cv2.xfeatures2d.SIFT_create()
    detect = cv2.xfeatures2d.SIFT_create()    
    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)
    fm = cv2.FlannBasedMatcher(flann_params, {})
    extract_bow = cv2.BOWImgDescriptorExtractor(extract, fm)
    extract_bow.setVocabulary(vocab)
    return extract_bow.compute(i, detect.detect(i))
    
def predict(image, vocabulary, trained_svm):
    """
    Taken from "OpenCV Computer Vision with Python" by Joseph Howse.  
    """
    f = bow_features(image, vocabulary)
    p = trained_svm.predict(f)
    return p
        

img = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories/accordion/image_0001.jpg"
file_loc = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories"
des_loc = "/home/centraltendency/Udacity/computer_vision_capstone/descriptors/descriptor"
pickle_loc = "/home/centraltendency/Udacity/computer_vision_capstone/training/BOW_pickle"
test_loc = "/home/centraltendency/Udacity/computer_vision_capstone/training/BOW_test"
train_loc = "/home/centraltendency/Udacity/computer_vision_capstone/training/BOW_train"
y_encoded = "/home/centraltendency/Udacity/computer_vision_capstone/training/BOW_train1"
bow_flann = "/home/centraltendency/Udacity/computer_vision_capstone/training/BOW_flann"
bow_1000clusters = "/home/centraltendency/Udacity/computer_vision_capstone/training/BOW_1000"

X, y = load_data(file_loc)

#le = preprocessing.LabelEncoder()
#le.fit_transform(labels)
#X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size = 0.1, random_state = 7)

X_train, X_test, y_train, y_test = split_data(X, y)

BOW_train = create_descriptors(X_train, 1000)
# BOW_test = create_descriptors(X_test, vocabulary_size)

#fileObject = open(pickle_loc, "r")
#vocabulary = pickle.load(fileObject)

vocabulary = create_codebook(BOW_train, bow_1000clusters)
vocabulary = get_vocabulary(bow_1000clusters)

train_data = create_flann_imgDescriptor(vocabulary, X_train)
test_data = create_flann_imgDescriptor(vocabulary, X_test)

# train svm with built in function

#svm = cv2.ml.SVM_create()
#svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(y_train))

predictions = []
for f in X:
    p = predict(f, vocabulary)
    predictions.extend(p)

# Train SVM
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


#xarray = np.asarray(train_data)
#y = y_train
#xtestarray = np.asarray(X_test)
#
#nsamples, nx, ny = xarray.shape
#d_dataset = xarray.reshape((nsamples, nx*ny))
#x = d_dataset
#nsamples, nx, ny = xtestarray.shape
#d_dataset = xtestarray.reshape((nsamples, nx*ny))
#xtest = d_dataset


pipeline = Pipeline([
        ('clf', SVC(kernel='rbf', gamma = 0.01, C = 100))
    ])
parameters = {
    'clf__gamma': (0.001, 0.01, 0.1, 1, 10, 100),
    'clf__C': (0.001, 0.01, 0.1, 1, 10, 100),
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs = -1, verbose = 1, scoring = 'accuracy')
grid_search.fit(np.array(train_data), np.array(y_train))
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])
    predictions = grid_search.predict(np.array(test_data))
    print classification_report(y_test, predictions)
    
clf = SVC(kernel = 'rbf', gamma = 10, C = 100)
clf.fit(train_data, y_train)
score = clf.score(test_data, y_test)

#predictions_1 = clf.predict(test_data)
#predictions_2 = clf.predict(train_data)
#predictions_1 == y_test
#predictions_2 == y_train

## Testing

example = predict(img, vocabulary, clf)