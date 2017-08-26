# -*- encoding:utf-8 -*-
"""
    Author:
    Name;

    Version:
"""
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time


#加载数据
t = time.time()
df_FallibilitySample = joblib.load('./model/FallibilitySample.df')
df_FallibilitySample = df_FallibilitySample.append(joblib.load('./model/FallibilitySample1.df'))
df_FallibilitySample = df_FallibilitySample.append(joblib.load('./model/FallibilitySample2.df'))
df_FallibilitySample = df_FallibilitySample.append(joblib.load('./model/FallibilitySample3.df'))
df_FallibilitySample = df_FallibilitySample.append(joblib.load('./model/FallibilitySample4.df'))
df_FallibilitySample = df_FallibilitySample.append(joblib.load('./model/FallibilitySample5.df'))
df_FallibilitySample = df_FallibilitySample.append(joblib.load('./model/FallibilitySample6.df'))
df_FallibilitySample = df_FallibilitySample.append(joblib.load('./model/FallibilitySample7.df'))
df_FallibilitySample = df_FallibilitySample.append(joblib.load('./model/FallibilitySample8.df'))
df_FallibilitySample = df_FallibilitySample.append(joblib.load('./model/FallibilitySample9.df'))
df_FallibilitySample = df_FallibilitySample.append(joblib.load('./model/FallibilitySample10.df'))
data = df_FallibilitySample
print data.shape
data = data.drop_duplicates()
print data.shape
print data.iloc[:, -1].value_counts()
#查看位置分布
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
mpl.rcParams['font.sans-serif'] = u'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
plt.figure(facecolor='w')
plt.scatter(data.__getitem__('aver_length'), data.__getitem__('Assignment'), c=pd.Categorical(data.iloc[:, -1]).codes, s=30,  marker='o', cmap=cm_dark)
plt.grid(b=True, ls=':')
plt.xlabel('aver_length', fontsize=14)
plt.ylabel('Assignment', fontsize=14)
plt.title(u'易误判样本点的分布情况', fontsize=18,)
# plt.savefig('1.png')
plt.show()


#对这些误判的点再次分类（SVM），观察效果
# clf = SVC(kernel='rbf', random_state=1)
clf = joblib.load('./model/clf_svm_13_v1.model')


x = data.iloc[:, 0:-1]
y = pd.Categorical(data.iloc[:, -1]).codes
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print confusion_matrix(y_test, y_pred)
print classification_report(y_test, y_pred)
print time.time() - t