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

def run():
    #
    np.set_printoptions(threshold='nan', linewidth=600, suppress=True)
    data_g = pd.read_csv('./imitateData_good.csv', header=0)
    data_b = pd.read_csv('./imitateData_bad.csv', header=0)
    #print np.array(data_g.iloc[0:3, :])
    data_g['label'] = 1
    #print np.array(data_g.iloc[0, :].reshape(-1))
    data_b['label'] = 0
    data = data_b.append(data_g)
    #
    x = data.iloc[:, 0:-1]
    y = data.iloc[:, -1]
    #print x.shape, y.shape, y
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=8, min_samples_split=8)
    clf.fit(x_train, y_train.ravel())
    #
    print '模型评分：', clf.score(x_train, y_train)
    y_hat = clf.predict(x_test)
    result = (y_hat == np.array(y_test))
    print '预测精度：', np.mean(result)

    #
    depth = np.arange(1, 12)
    split_leaf = np.arange(2, 13)
    predictions = []
    for d in depth:
        for sl in split_leaf:
            clf_param = DecisionTreeClassifier(criterion='entropy', max_depth=d, min_samples_split=sl, min_samples_leaf=sl)
            clf_param.fit(x_train, y_train)
            y_hat_param = clf_param.predict(x_test)
            result_param = (y_hat_param == y_test)
            predictions.append(np.mean(result_param))
    print len(predictions), np.argmax(predictions)
    depth_index = np.argmax(predictions) / len(depth) + 1
    depth_excellent = depth.__getitem__(depth_index-1)
    print '最优深度：', depth_excellent
    split_leaf_index = np.argmax(predictions) % len(depth)
    sl_excellent = split_leaf.__getitem__(split_leaf_index-1)
    print '最优sl: ', sl_excellent

    #按照最优参数建模
    clf_excellent = DecisionTreeClassifier(criterion='entropy', max_depth=depth_excellent, min_samples_leaf=sl_excellent, min_samples_split=sl_excellent)
    clf_excellent.fit(x_train, y_train)
    y_hat_excellent = clf_excellent.predict(x_test)
    result_excellent = (y_hat_excellent == y_test)
    print '最优参数下的模型精度：', np.mean(result_excellent)
if __name__ == '__main__':
    run()