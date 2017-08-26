# -*- encoding: utf-8
"""
    Author:
    Name:
    Version:
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

data = pd.read_csv('../data/train.csv', header=0)
data['label'] = data['flag']
data.__delitem__('flag')
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print confusion_matrix(y_test, y_pred)
print classification_report(y_test, y_pred)