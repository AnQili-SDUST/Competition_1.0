# -*-encoding:utf-8 -*-
"""
    Author:
    Name:
    Version:
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
import matplotlib as mpl
import matplotlib.pyplot as plt
def run():
    #
    pd.set_option('display.width', 800, 'max_rows', 10000, 'max_columns', 100)
    np.set_printoptions(threshold='nan', linewidth=800, suppress=True)
    data = pd.read_csv('./data/train.csv', header=0)
    data.dropna(inplace=True)
    print data.shape
    data['label'] = data['flag']
    data.__delitem__('flag')
    x = data.iloc[:, 0:-1]
    y = pd.Categorical(data.iloc[:, -1]).codes
    #
    fs = SelectKBest(chi2, k=4)
    fs.fit(x, y)
    idx = fs.get_support(indices=True)
    print 'fs.get_support() = ', idx
    x = x[idx]
    x = x.values  # 为下面使用方便，DataFrame转换成ndarray
    print 'BestFeatureName：', data[idx].columns
    # print data.__getitem__(idx)#---手动添加坐标轴名称
    x0_label, x1_label, x2_label, x3_label = data[idx].columns
    # x1_label = idx[0]
    # x2_label = idx[1]
    title = u'4-20数据特征选择'
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.scatter(x[:, 0], x[:, 1], s=30, c=y, marker='o', cmap=cm_dark)
    plt.grid(b=True, ls=':')
    plt.xlabel(x0_label, fontsize=14)
    plt.ylabel(x1_label, fontsize=14)
    plt.title(title, fontsize=18)
    # plt.savefig('1.png')
    plt.show()



if __name__ == '__main__':
    run()