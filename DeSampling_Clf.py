# -*-coding: utf-8 -*-
"""
    Author:
    Name:
    Version:
"""
import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from  sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import time
def RunMain():
    # 设置控制台显示参数
    np.set_printoptions(threshold='nan', linewidth=800, suppress=True)
    pd.set_option('display.width', 800, 'max_rows', 10000, 'max_columns', 100)
    # 读入数据
    data = pd.read_csv('./data/train_1.csv', header=0)
    data.dropna(inplace=True)
    data['label'] = data['flag']
    data.__delitem__('flag')
    print '总样本规模', data.shape
    # 随机去掉稠密部分的一部分样本点
    #  data[(data.length< 50000) & (data.emp< 75000)]
    print '稀疏后的样本规模:'
    # 选取待降采样的样本(正例样本)
    data_n = data[data.label == 'n']
    print '正例样本规模', data_n.shape
    # 选取负例样本
    data_pd = data[(data.label == 'p') | (data.label == 'd')]
    print '负例样本规模', data_pd.shape
    # 得到均衡的子数据集
    data_sub = RepttitionSamplingAndGetSubData(data_n, 2000, data_pd)
    print '均衡后的子样本规模', data_sub.shape
    print '子样本中各类样本的个数', data_sub.iloc[:, -1].value_counts()
    # 设定x, y
    x = data_sub.iloc[:, 0:-1]
    y = pd.Categorical(data_sub.iloc[:, -1]).codes
    # 显示子样本集的数据分布
    GraphDistribute(x, y, data_sub)
    # 划分训练、测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    x_test = data.iloc[300:1000, 0:-1]  #----****测试单个随机对单个随机的效果
    y_test = pd.Categorical(data.iloc[300:1000, -1]).codes
    # 构建分类模型进行分类,输出分类结果
    ClfModel(x_train, y_train, x_test, y_test, x, y)
    # ###临时构建一个使用全体集训练的svc模型
    # alpha = np.logspace(-10, 30, 100)
    # clf = Pipeline([('sc', StandardScaler()),
    #                 ('poly', PolynomialFeatures(degree=2)),
    #                 ('clf1', RandomizedSearchCV(SVC(kernel='rbf', decision_function_shape='ovr'),
    #                 param_distributions = {'C': alpha, 'gamma': alpha}))])
    # clf.fit(data.iloc[:, 0:-1], pd.Categorical(data.iloc[:, -1]).codes)
    # joblib.dump(clf, './model/clf_svm_13_v1-1_4w.model')
    # 总耗时
    print '总耗时：', time.time() - t





# 得到一个均衡的子数据集（对正例进行降采样并与负例合成一个子样本集）
def RepttitionSamplingAndGetSubData(data_n, number, data_pd):
    data_n_sampled = []
    for i in range(number):
        data_n_sampled.append(data_n.iloc[random.randint(0, len(data_n)-1), :])
    data_n_sampled = pd.DataFrame(data_n_sampled)
    data_sub = data_n_sampled.append(data_pd)
    return data_sub

# 可视化显示子数据集的样本分布(2-BestFeature)
def GraphDistribute(x, y, data_sub):
    fs = SelectKBest(chi2, k=2)
    fs.fit(x, y)
    idx = fs.get_support(indices=True)
    print 'fs.get_support() = ', idx
    x_2Bf = x[idx]
    x_2Bf = x_2Bf.values  # 为下面使用方便，DataFrame转换成ndarray
    # print data_sub.__getitem__(idx)  # ---手动添加坐标轴名称
    x1_label = idx[0]
    x2_label = idx[1]
    title = u'4-20数据特征选择'
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.scatter(x_2Bf[:, 0], x_2Bf[:, 1], s=30, c=y, marker='o', cmap=cm_dark)
    plt.grid(b=True, ls=':')
    plt.xlabel(x1_label, fontsize=14)
    plt.ylabel(x2_label, fontsize=14)
    plt.title(title, fontsize=18)
    # plt.savefig('1.png')
    plt.show()

# 构建分类模型进行分类
def ClfModel(x_train, y_train, x_test, y_test, x, y):
    # alpha = np.logspace(-20, 20, 200)
    # clf = Pipeline([('sc', StandardScaler()),
    #                 ('poly', PolynomialFeatures(degree=3)),
    #                 ('clf1', GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovr'),
    #                               param_grid={'C': alpha, 'gamma': alpha}, cv=5))])
    clf = DecisionTreeClassifier(criterion='entropy')
    # depth = range(2, 13)
    # split = range(2, 15)
    # leaf = range(1, 14)
    # clf = GridSearchCV(DecisionTreeClassifier(criterion='entropy'), param_grid={'max_depth':depth, 'min_samples_split':split, 'min_samples_leaf':leaf}, cv=5)
    clf.fit(x_train, y_train.ravel())
    print '模型训练，已耗时：', time.time() - t
    #保存模型
    joblib.dump(clf, './model/clf_svm_13_v4.model')
    y_pred = clf.predict(x_test)
    print confusion_matrix(y_test, y_pred)
    print classification_report(y_test, y_pred)







if __name__ == '__main__':
    t = time.time()
    RunMain()