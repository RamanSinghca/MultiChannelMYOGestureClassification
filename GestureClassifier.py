'''
Created on Oct 28, 2016
Classifier models for Gesture Recognition

@author: ramansingh, b4s79@unb.ca
'''

import os
import time
import pandas
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.lda import LDA
from sklearn import decomposition
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression



userhome = os.path.expanduser('~')
filename='otherFeatureTrain'
Data = userhome +r'/Desktop/Data/thalmicMyoData/trainingData/{fname}.csv'.format(fname=filename)

allData = pandas.read_csv(Data, header=None);
dataset = allData.values
np.random.shuffle(dataset)
X=dataset[:,0:allData.shape[1]-1]
Y= dataset[:,allData.shape[1]-1]

#  K for cross fold validation
kFold=5

#if 0, no reduction in features. 
# if more than 0 , reduce features to specified components.
pca_components=0

# reduced the number of features to #pca_components
if pca_components>0:
    pca = decomposition.PCA(pca_components)
else:
    #If no argument provided, original features stays
    pca = decomposition.PCA()
pca.fit(X)
X=pca.transform(X)


def ClassifySVM():
    kf = KFold(len(X), n_folds=kFold)
    for train, test in kf:
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        clf = svm.SVC(C=50.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ova', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=True, random_state=None, shrinking=True,
        tol=0.001, verbose=False)    
        Y_predicted= clf.fit(X_train, Y_train).predict(X_test)
        print()
        print (metrics.classification_report(Y_test, Y_predicted))
        print()
        print ("Confusion matrix")
        print (metrics.confusion_matrix(Y_test, Y_predicted))

def ClassifyLDA():
    kf = KFold(len(X), n_folds=kFold)
    for train, test in kf:
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        clf = LDA()
        Y_predicted= clf.fit(X_train, Y_train).predict(X_test)
        print()
        print (metrics.classification_report(Y_test, Y_predicted))
        print()
        print ("Confusion matrix")
        print (metrics.confusion_matrix(Y_test, Y_predicted))

def ClassifyLogisticsRegress():
    kf = KFold(len(X), n_folds=kFold)
    for train, test in kf:
        X_train, X_test, Y_train, Y_test = X[train], X[test], Y[train], Y[test]
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        # make predictions
        Y_predicted = model.predict(X_test)
        # summarize the fit of the model
        print()
        print (metrics.classification_report(Y_test, Y_predicted))
        print()
        print ("Confusion matrix")
        print (metrics.confusion_matrix(Y_test, Y_predicted))



start_time = time.clock()
ClassifyLDA()
#ClassifyLogisticsRegress()
#ClassifySVM()
print ('Time Taken to train and Validate the Model ' , time.clock() - start_time, "seconds")





