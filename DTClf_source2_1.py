# -*- encoding:utf-8
"""
    Author:
    Name:
    Version:
    Date:
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.semi_supervised import label_propagation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
import time
def run():
    #
    startTime = time.time()
    np.set_printoptions(threshold='nan', linewidth=600, suppress=True)
    data = pd.read_csv('./exampleData2_1.csv', header=0)
    print data.shape
    data = data.dropna(axis=0)
    print data.shape
    data['label'] = data.__getitem__('flag')
    data.__delitem__('flag')

    #
    x = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    # y = pd.Categorical(y).codes
    #print y.unique()
    #print y.value_counts()
    #print x.shape, y.shape
    #print np.unique(y)

    # PCA
    pca = PCA(n_components=10, whiten=True, random_state=12)
    x = pca.fit_transform(x)
    # print '各方向方差', pca.explained_variance_
    # print '各方向所占方差比例', pca.explained_variance_ratio_

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)

    # clf = DecisionTreeClassifier(criterion='gini', max_depth=2, min_samples_leaf=8, min_samples_split=8)
    # clf.fit(x_train, y_train.ravel())
    # print y_test.value_counts()
    #print clf.classes_

    #
    #print '初始模型score：', clf.score(x_train, y_train)
    # y_hat = clf.predict(x_test)
    # result = (y_hat == np.array(y_test))
    #print '初始预测精度：', np.mean(result)

    #
    depth = np.arange(1, 15)
    split_leaf = np.arange(2, 16)
    predictions = []
    for d in depth:
        for sl in split_leaf:
            # clf_param = DecisionTreeClassifier(criterion='entropy', max_depth=d, min_samples_split=sl, min_samples_leaf=sl)
            clf_param = Pipeline([
                ('sc', StandardScaler()),
                ('poly', PolynomialFeatures(degree=4)),
                ('clf', DecisionTreeClassifier(criterion='entropy', max_depth=d, min_samples_leaf=sl, min_samples_split=sl))])
            clf_param.fit(x_train, y_train)
            y_hat_param = clf_param.predict(x)
            result_param = (y_hat_param == y)
            predictions.append(np.mean(result_param))
    #print len(predictions), np.argmax(predictions)
    depth_index = np.argmax(predictions) / len(depth) + 1
    depth_excellent = depth.__getitem__(depth_index-1)
    print '参数调试：'
    print '最优深度：', depth_excellent
    split_leaf_index = np.argmax(predictions) % len(depth)
    sl_excellent = split_leaf.__getitem__(split_leaf_index-1)
    print '最优sl: ', sl_excellent


    #按照最优参数建模
    clf_excellent = Pipeline([
        ('sc', StandardScaler()),
        ('poly', PolynomialFeatures(degree=4)),
        ('clf', DecisionTreeClassifier(criterion='entropy', max_depth=depth_excellent, min_samples_leaf=sl_excellent, min_samples_split=sl_excellent))])
    # clf_excellent = DecisionTreeClassifier(criterion='entropy', max_depth=depth_excellent, min_samples_leaf=sl_excellent, min_samples_split=sl_excellent)
    clf_excellent.fit(x_train, y_train)
    y_hat_excellent = clf_excellent.predict(x)
    result_excellent = (y_hat_excellent == y)
    # #print '调参后模型预测精度：', np.mean(result_excellent)
    #
    print confusion_matrix(y, y_hat_excellent, labels=clf_excellent.classes_)
    print classification_report(y, y_hat_excellent)
    endTime = time.time()
    print '总耗时：', endTime-startTime
if __name__ == '__main__':
    run()