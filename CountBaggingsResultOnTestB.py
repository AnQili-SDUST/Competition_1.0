# -*- encoding: utf-8 -*-
"""
    Author:
    Name:
    Describe:
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.externals import joblib
def run():
    #数据准备
    data, data_n, data_pd = DataPrepare()
    print '总样本规模：', data.shape, data.iloc[:, -1].value_counts()
    print '正例样本规模', data_n.shape
    print '负例样本规模', data_pd.shape

    # 测试集准备
    data_B = TestDataPrepare(data)
    print data_B.head()

    # #构建Bagging模型组
    # # clf_Ls_BG_DT_ms65 = []
    # # for i in range(9):
    # #     data_sub = RepttitionGetSubData(data_n, 0.65, data_pd, 0.95)
    # #     x = data_sub.iloc[:, 0:-1]
    # #     y = pd.Categorical(data_sub.iloc[:, -1]).codes
    # #     clf = BaggingClassifier(DecisionTreeClassifier(criterion='entropy'), n_estimators=99, max_samples=0.65, max_features=1.0, bootstrap_features=True)
    # #     clf.fit(x, y)
    # #     clf_Ls_BG_DT_ms65.append(clf)
    # # joblib.dump(clf_Ls_BG_DT_ms65, './model/clf_Ls_BG_DT_ms65.model')
    #
    # # 预测testB
    # clf_Ls = joblib.load('./model/clf_Ls_BG_DT_ms65.model')
    # BGsResult_Ls = []
    # for i in range(len(clf_Ls)):
    #     clf = clf_Ls[i]
    #     y_pred = clf.predict(data_B.iloc[:, 0:-1])
    #     result = pd.DataFrame(data_B.id)
    #     result['y_pred'] = y_pred
    #     BGsResult_Ls.append(result)
    # joblib.dump(BGsResult_Ls, './model/BGsReslut_DataFrame_LS.list')
    #
    # #统计结果
    # ResLs = joblib.load('./model/BGsReslut_DataFrame_LS.list')
    # BGsCSV_Ls = []
    # for i in range(len(ResLs)):
    #     res = ResLs[i]
    #     res = res[(res.y_pred == 0) | (res.y_pred == 2)]
    #     res_d = res[res.y_pred == 0]
    #     res_p = res[res.y_pred == 2]
    #     res_d['y_pred_label'] = 'd'
    #     res_p['y_pred_label'] = 'p'
    #     res = res_p.append(res_d)
    #     res.__delitem__('y_pred')
    #     res.to_csv('./result/BGsResulst_alpha_%d.csv'%(i), index=False)

    # 单个模型 训练，预测，写出
    clf = BaggingClassifier(DecisionTreeClassifier(criterion='entropy'), n_estimators=99, max_samples=0.65, max_features=1.0, bootstrap_features=True)
    clf.fit(data.iloc[:, 0:-1], pd.Categorical(data.iloc[:, -1]).codes)
    y_pred = clf.predict(data_B.iloc[:, 0:-1])
    result = pd.DataFrame(data_B.id)
    result['y_pred'] = y_pred
    result = result[(result.y_pred == 0) | (result.y_pred == 2)]
    result_d = result[result.y_pred == 0]
    result_p = result[result.y_pred == 2]
    result_d['y_pred_label'] = 'd'
    result_p['y_pred_label'] = 'p'
    result = result_p.append(result_d)
    result.__delitem__('y_pred')
    result.to_csv('./result/BGsResulst_alpha.csv', index=False)


#数据准备
def DataPrepare():
    # 读入数据
    data = pd.read_csv('./data/train_4_30.csv', header=0)
    data.dropna(inplace=True)
    # 去掉新增特征
    data_oldCols = pd.read_csv('./data/train.csv', header=0)
    data_oldCols.dropna(inplace=True)
    data = data.loc[:, data.columns.isin(data_oldCols.columns)]
    #
    data['label'] = data['flag']
    data.__delitem__('flag')
    #筛出整理样本
    data_n = data[data.label == 'n']
    #筛取负例样本
    data_pd = data[(data.label == 'p') | (data.label == 'd')]
    return data, data_n, data_pd

def RepttitionGetSubData(data_n, u, data_pd, v):
    data_n, data_n_noUse = train_test_split(data_n, train_size=u, random_state=1)
    data_pd , data_pd_noUse = train_test_split(data_pd, train_size=v, random_state=1)
    data_sub = data_n.append(data_pd)
    return data_sub

def TestDataPrepare(data_template):#为保持训练与测试的数据各维相同，此处以上面处理好的训练集data为data_template模板
    data = pd.read_csv('./data/testB.csv', header=0)
    data.dropna(inplace=True)
    data_backToOldCols = data.loc[:, data.columns.isin(data_template.columns)]
    data_backToOldCols['id'] = data.id
    return data_backToOldCols


if __name__ == "__main__":
    # 设置控制台显示参数
    warnings.filterwarnings('ignore')
    np.set_printoptions(threshold='nan', linewidth=800, suppress=True)
    pd.set_option('display.width', 800, 'max_rows', 10000, 'max_columns', 100)
    run()