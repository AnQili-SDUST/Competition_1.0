# -*- encoding: utf-8-*-
"""
    Author:
    Name:
    Version:
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import time

#
print 'starttime:', time.strftime('%y/%m/%d-%H:%M:%S', time.localtime())
np.set_printoptions(precision=3, threshold='nan', linewidth=500)
#
data = pd.read_csv('./exampleData2_1.csv', header=0)
data = data.dropna()
print data.shape
data['label'] = data['flag']
data.__delitem__('flag')
y = pd.Categorical(data.iloc[:, -1]).codes
x = data.iloc[:, 0:33]
#
pca = PCA(n_components=28, whiten=True, random_state=0)
x = pca.fit_transform(x)
#
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=12)
alpha = np.logspace(-2, 2, 20)
clf = Pipeline([('sc', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=2)),
                   ('clf1', GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovo', class_weight={0: 8, 1: 1, 2: 10}), param_grid={'C': alpha, 'gamma': alpha}, cv=5))])
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
#
#print y_pred
#print y_test.shape, y_pred.shape
print confusion_matrix(y_test, y_pred)
print classification_report(y_test, y_pred)
print 'endtime:', time.strftime('%y/%m/%d-%H:%M:%S', time.localtime())