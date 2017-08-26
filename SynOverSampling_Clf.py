# -*- encoding:  utf-8 -*-
"""
    Author:
    Name:
    Version:
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest,chi2
from SMOTE import Smote
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from Competition.DeSampling_Clf import RepttitionSamplingAndGetSubData

# 设置控制台显示参数
np.set_printoptions(threshold='nan', linewidth=800, suppress=True)
pd.set_option('display.width', 800, 'max_rows', 10000, 'max_columns', 100)
#准备数据
data = pd.read_csv('./data/train.csv', header=0)
data['label'] = data['flag']
data.__delitem__('flag')
print data['label'].value_counts()
#为d类做smote
data_d = data[data.label == 'd']
x = data_d.iloc[:, 0:-1]
x = np.array(x)
s = Smote(x, 100, 4)
data_d_s = s.over_sampling()
data_d_s = pd.DataFrame(data_d_s)
data_d_s['label'] = 'd'
data_d_s.columns = data_d.columns
#为p类做smote
data_p = data[data.label == 'p']
x = data_p.iloc[:, 0:-1]
x = np.array(x)
s = Smote(x, 100, 4)
data_p_s = s.over_sampling()
data_p_s = pd.DataFrame(data_p_s)
data_p_s['label'] = 'p'
data_p_s.columns = data_p.columns
#组合成新的数据集
data = data.append(data_d_s)
data = data.append(data_p_s)
print data_d_s.head()
print data_p_s.head()
print data[data.label == 'd'].index



#结合降采样分析效果（可注释掉）
data_n = data[data.label == 'n']
data_pd = data[(data.label == 'p') | (data.label == 'd')]
data = RepttitionSamplingAndGetSubData(data_n, 4000, data_pd)

#训练模型
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
print data['label'].value_counts()
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print confusion_matrix(y_test, y_pred)
print classification_report(y_test, y_pred)




# #用最初的4w的数据集去检验该模型
# data = pd.read_csv('./data/train.csv', header=0)
# data['label'] = data['flag']
# data.__delitem__('flag')
# x = data.iloc[:, 0:-1]
# y = data.iloc[:, -1]
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# print confusion_matrix(y_test, y_pred)
# print classification_report(y_test, y_pred)