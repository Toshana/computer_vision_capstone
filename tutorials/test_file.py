# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:26:28 2017

@author: centraltendency
"""
# >>> sys.path.remove('/usr/lib/python2.7/dist-packages')


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
    
def create_descriptors(x, vocabulary_size):
    BOW_train = cv2.BOWKMeansTrainer(vocabulary_size)
    sift = cv2.xfeatures2d.SIFT_create()
    for f in x:
        kp, des = sift.detectAndCompute(f, None)
        BOW_train.add(des)
    return BOW_train
        
def create_codebook(BOW_trainer, file_name):
    vocabulary = BOW_train.cluster()
    fileObject = open(file_name, 'wb')
    pickle.dump(vocabulary, fileObject)
    fileObject.close()
    return vocabulary
    
def get_vocabulary(file_location):
    fileObject = open(file_location, "r")
    vocabulary = pickle.load(fileObject)
    return vocabulary
    
train_labels = []
def create_imgDescriptor(BOW, x):
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
     
def bow_features(image_path, vocab):
    extract = cv2.xfeatures2d.SIFT_create()
    detect = cv2.xfeatures2d.SIFT_create()    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    extract_bow = cv2.BOWImgDescriptorExtractor(extract, bf)
    extract_bow.setVocabulary(vocab)
    im = cv2.imread(image_path, 0)
    return extract_bow.compute(im, detect.detect(im))
    
def predict(image, vocabulary):
    f = bow_features(image, vocabulary)
    p = svm.predict(f)
    print image, "\t", p[1][0][0]
    return p
        

img = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories/accordion/image_0001.jpg"
file_loc = "/home/centraltendency/Udacity/computer_vision_capstone/ObjectCategories"
des_loc = "/home/centraltendency/Udacity/computer_vision_capstone/descriptors/descriptor"
pickle_loc = "/home/centraltendency/Udacity/computer_vision_capstone/training/BOW_pickle"
test_loc = "/home/centraltendency/Udacity/computer_vision_capstone/training/BOW_test"
train_loc = "/home/centraltendency/Udacity/computer_vision_capstone/training/BOW_train"
y_encoded = "/home/centraltendency/Udacity/computer_vision_capstone/training/BOW_train1"

X, y = load_data(file_loc)

#le = preprocessing.LabelEncoder()
#le.fit_transform(labels)
#X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size = 0.1, random_state = 7)

X_train, X_test, y_train, y_test = split_data(X, y)

BOW_train = create_descriptors(X_train, vocabulary_size)
# BOW_test = create_descriptors(X_test, vocabulary_size)

#fileObject = open(pickle_loc, "r")
#vocabulary = pickle.load(fileObject)

vocabulary = create_codebook(BOW_train, y_encoded)
vocabulary = get_vocabulary(y_encoded)

train_data = create_imgDescriptor(vocabulary, X_train)
test_data = create_imgDescriptor(vocabulary, X_test)

# train svm with built in function

svm = cv2.ml.SVM_create()
svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(y_train))

predictions = predict(img, vocabulary)

# Train SVM
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
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

clf = SVC()
clf.fit(np.array(train_data), np.array(y_train))
predictions = clf.predict(np.array(test_data))


pipeline = Pipeline([
        ('clf', SVC(kernel='rbf', gamma = 0.01, C = 100))
    ])
print x.shape
parameters = {
    'clf__gamma': (0.001, 0.01, 0.1, 1, 10, 100, 1000),
    'clf__C': (0.001, 0.01, 0.1, 1, 10, 100, 1000),
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
    
## Naive Bayes
    
from sklearn.naive_bayes import MultinomialNB
pipeline = Pipeline([
        ('bayes', MultinomialNB(alpha = 1.0, fit_prior = True, class_prior = None))
        ])
parameters = {
    'bayes__alpha': (0.0001, 0.01, 0.1, 1, 10, 100, 1000)
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
