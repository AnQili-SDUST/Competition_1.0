# -*-encoding:utf-8 -*-
"""
    Author:
    Name:
    Version:
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import random
from sklearn.neighbors import KNeighborsClassifier
def run():
    # 设置控制台显示参数
    np.set_printoptions(threshold='nan', linewidth=800, suppress=True)
    pd.set_option('display.width', 800, 'max_rows', 10000, 'max_columns', 100)
    # 读入数据
    data = pd.read_csv('./data/train_4_30.csv', header=0)
    data.dropna(inplace=True)
    #去掉新增特征
    data_oldCols = pd.read_csv('./data/train.csv', header=0)
    data = data.loc[:, data.columns.isin(data_oldCols.columns)]
    #
    data['label'] = data['flag']
    data.__delitem__('flag')
    print '总样本规模', data.shape
    print data.label.value_counts()
    #正例降采样50%:
    # 选取待降采样的样本(正例样本)
    data_n = data[data.label == 'n']
    # 选取负例样本
    data_pd = data[(data.label == 'p') | (data.label == 'd')]
    # 按比例选取样本子集
    # data_sub = RepttitionSamplingAndGetSubData(data_n, 0.85, data_pd, 0.95)
    data_sub = data
    print '选取的样本的规模：', data_sub.shape
    print data_sub.loc[:, 'label'].value_counts()
    x = data_sub.iloc[:, 0:-1]
    y = pd.Categorical(data_sub.iloc[:, -1]).codes
    #
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1)
    #
    clf = BaggingClassifier(DecisionTreeClassifier(criterion='entropy'), n_estimators=89, max_samples=0.65, max_features=1.0, bootstrap_features=True)
    # clf = BaggingClassifier([])
    # clf = AdaBoost stingClassifier(learning_rate=0.2)
    # clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train.ravel())

    #
    y_pred = clf.predict(x_test)
    print confusion_matrix(y_test, y_pred)
    print classification_report(y_test, y_pred)


def RepttitionSamplingAndGetSubData(data_n, u, data_pd, v):
    # data_n_sampled = []
    # for i in range(number):
    #     data_n_sampled.append(data_n.iloc[random.randint(0, len(data_n)-1), :])
    # data_n_sampled = pd.DataFrame(data_n_sampled)
    data_n, data_n_noUse = train_test_split(data_n, train_size=u, random_state=1)
    data_pd , data_pd_noUse = train_test_split(data_pd, train_size=v, random_state=1)
    data_sub = data_n.append(data_pd)
    return data_sub

if __name__ == "__main__":
    run()