# -*- encoding:utf-8 -*-
"""
    Author:
    Name:
    Version:
"""
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

#加载特殊类型数据（正负例样本不均衡）
data = pd.read_csv('../data/train.csv', header=0)
data['label'] = data['flag']
data.__delitem__('flag')
x, y = data.iloc[:, 0:-1], data.iloc[:, -1]
#加载普通数据（各类样本数量均衡）
# data = load_iris()
# x, y = data.data, data.target

y = pd.Categorical(y).codes
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)
data_train = xgb.DMatrix(x_train, label=y_train)
data_test = xgb.DMatrix(x_test, label=y_test)
watch_list = [(data_test, 'eval'), (data_train, 'train')]
param = {'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}


## 使用xgboost--xgbClf
# clf = xgb.XGBClassifier(n_estimators=1)
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
##使用xgboost--train()
bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)
y_pred = bst.predict(data_test)
##使用普通决策树
# clf = DecisionTreeClassifier(criterion='entropy')
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)



print confusion_matrix(y_test, y_pred)
print classification_report(y_test, y_pred)