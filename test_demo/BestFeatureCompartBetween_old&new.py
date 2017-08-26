# -*- encoding: utf-8 -*-
"""
    Author:
    Name:
    Describe:
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
#old集
data_old = pd.read_csv('../data/train.csv', header=0)
data_old['label'] = data_old['flag']
data_old.__delitem__('flag')
x = data_old.iloc[:, 0:-1]
y = data_old.iloc[:, -1]
fs = SelectKBest(chi2, 10)
fs.fit(x, y)
print data_old.shape
print y.value_counts()
print list(data_old[fs.get_support(indices=True)].columns)
#old_1集
data_middle = pd.read_csv('../data/train_1.csv', header=0)
data_middle.dropna(inplace=True)
data_middle['label'] = data_middle['flag']
data_middle.__delitem__('flag')
x = data_middle.iloc[:, 0:-1]
y = data_middle.iloc[:, -1]
fs = SelectKBest(chi2, 10)
fs.fit(x, y)
print data_middle.shape
print y.value_counts()
print list(data_middle[fs.get_support(indices=True)].columns)
#new集
data_new = pd.read_csv('../data/train_4_30.csv', header=0)
data_new.dropna(inplace=True)
data_new['label'] = data_new['flag']
data_new.__delitem__('flag')
x = data_new.iloc[:, 0:-1]
y = data_new.iloc[:, -1]
fs = SelectKBest(chi2, 11)
fs.fit(x, y)
print data_new.shape
print y.value_counts()
print list(data_new[fs.get_support(indices=True)].columns)

# #从train_4_30中过滤掉新增的特征
# data = data_new.loc[:, data_new.columns.isin(data_old.columns)]
# print data.shape
# print list(data.columns)
# print list(data_old.loc[:, data.columns.isin(data_old.columns)].columns)
# print len(list(data_old.loc[:, data.columns.isin(data_old.columns)].columns))