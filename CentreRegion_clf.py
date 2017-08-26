# -*- encoding:utf-8 -*-
"""
    Author:
    Name:
    Version:
"""
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from Bagging_WeakClfs_Vote import VoteCompute
import warnings
def run():
    #加载数据
    data = pd.read_csv('./data/train.csv', header=0)
    data['label'] = data['flag']
    data.__delitem__('flag')
    #筛选核心区域数据
    data_centre = data[(data.emp < 2000) & (data.length < 5000) & (data.aver_length < 1200) & (data.Assignment < 140)]
    data_surround = data[(data.emp >= 2000) | (data.length >= 5000) | (data.aver_length >= 1200) | (data.Assignment >= 140) ]
    print data.shape
    print data_centre.shape
    print data_surround.shape

    #中心区域用svm_v4
    y_centre_test, y_centre_pred = CentreClf(data_centre)
    #周边区域用决策树投票
    y_surround_test, y_surround_pred = SurroundClf(data_surround)
    y_test = y_centre_test.append(y_surround_test)
    y_pred = y_centre_pred.append(y_surround_pred)
    print '中心区域：'
    print confusion_matrix(y_centre_test, y_centre_pred)
    print classification_report(y_centre_test, y_centre_pred)
    print '周围区域：'
    print confusion_matrix(y_surround_test, y_surround_pred)
    print classification_report(y_surround_test, y_surround_pred)
    print '总评：'
    print confusion_matrix(y_test, y_pred)
    print classification_report(y_test, y_pred)


def CentreClf(data_centre):
    x = data_centre.iloc[:, 0:-1]
    y = pd.Categorical(data_centre.iloc[:, -1]).codes
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    clf = joblib.load('./model/clf_svm_13_v1.model')
    y_pred = clf.predict(x)
    print '中心区域预测完毕...'
    return pd.DataFrame(y), pd.DataFrame(y_pred)

def SurroundClf(data_surround):
    x = data_surround.iloc[:, 0:-1]
    y = pd.Categorical(data_surround.iloc[:, -1]).codes
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    clf_Ls = joblib.load('./model/clf_Ls_DecisionTree_99.model')
    sample_pred = []
    for i in range(len(x)):
        sample_preds = []
        for j in range(len(clf_Ls)):
            clf = clf_Ls[j]
            sample_preds.append(int(clf.predict(x.iloc[i, :])))
        whoWin = VoteCompute(sample_preds)
        sample_pred.append(whoWin)
        print '周围区域第', i, '个样本预测完成...', i*1.0/len(x)*100, '%'
    return pd.DataFrame(y), pd.DataFrame(sample_pred)
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    run()