# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def extend(a, b):
    return 1.05*a-0.05*b, 1.05*b-0.05*a


if __name__ == '__main__':
    pca = False
    pd.set_option('display.width', 800, 'max_rows', 10000, 'max_columns', 800)
    data = pd.read_csv('exampleData2_1.csv', header=0)
    data.dropna(inplace=True)

    data['label'] = data['flag']
    data.__delitem__('flag')
    print data.head()
    data = data[data.iloc[:, -1].isin(['p', 'd'])]#******此处只选取了两类负例
    print data.head()
    x = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    if pca:
        pca = PCA(n_components=3, whiten=True, random_state=0)
        x = pca.fit_transform(x)
        print '各方向方差：', pca.explained_variance_
        print '方差所占比例：', pca.explained_variance_ratio_

    else:
        #fs = SelectKBest(chi2, k=20)
        fs = SelectPercentile(chi2, percentile=100)
        fs.fit(x, y)
        idx = fs.get_support(indices=True)
        print 'fs.get_support() = ', idx
        x = x[idx]



    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=True)),
        ('lr', LogisticRegressionCV(Cs=np.logspace(-3, 4, 8), cv=5, fit_intercept=False))
    ])
    model.fit(x, y)
    print '最优参数：', model.get_params('lr')['lr'].C_
    y_pred = model.predict(x)
    print confusion_matrix(y, y_pred, labels=model.classes_)
    print classification_report(y, y_pred)

