# -*- encoding: utf-8 -*-
"""
    Author:
    Name:
    Describe:
"""

import pandas  as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def run():
    data = pd.read_csv('./data/confuse/confuse_train.csv', header=0)
    print data.head()
    data.dropna(inplace=True)
    data['lebel'] = data.flag
    data.__delitem__('flag')
    data.__delitem__('filename')
    print data.head()
    x = data.iloc[:, 0:-1]
    y = pd.Categorical(data.iloc[:, -1]).codes
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    #
    # clf = AdaBoostRegressor(SVC(C=5, kernel='linear'))
    # clf = DecisionTreeClassifier('entropy')
    # C = np.linspace(0.5, 15, 5)
    # clf = GridSearchCV(SVC(C), param_grid={'C': C})
    clf = Pipeline([
        ('sc', StandardScaler()),
        ('poly', PolynomialFeatures(degree=9)),
        ('clf', SVC(C=4.6, kernel='rbf'))])
    clf.fit(x_train, y_train)
    # print clf.best_params_
    # joblib.dump(clf, './model/confuse_model.model')
    #
    # clf = joblib.load('./model/confuse_model.model')
    y_pred = clf.predict(x_test)
    print confusion_matrix(y_test, y_pred)
    print classification_report(y_test, y_pred)
if __name__ == "__main__":
    run()