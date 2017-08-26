# -*-encoding:utf-8 -*-
"""
    Author:
    Name:
    Version:
"""
import pandas as pd
from sklearn.externals import joblib
import random
import numpy as np
from sklearn.ensemble import BaggingClassifier
from SMOTE import Smote
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import time
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import warnings
from sklearn.model_selection import RandomizedSearchCV
def run():
    #读入并整理数据
    data = pd.read_csv('./data/train_exmp_combine_col50_index.csv', header=0)
    data.dropna(inplace=True)
    data['label'] = data['flag']
    data.__delitem__('flag')
    print data.shape
    #smote扩充负例样本点
    ##扩充p类样本
    data_p = data[data.label == 'p']
    print 'p类样本的初始规模', data_p.shape
    ##去掉靠近主要特征的边缘的样本----本次缩短比例为上次的0.8
    # data_p = data_p[(data_p.length < 8000) & (data_p.emp < 12000) & (data_p.aver_length < 4000) & (data_p.Assignment < 144)]
    # x = data_p.iloc[:, 0:-1]
    # y = data_p.iloc[:, -1]
    # s = Smote(np.array(x), 100, 4)
    # x_s = s.over_sampling()
    # x_s = pd.DataFrame(x_s)
    # x_s['label'] = 'p'
    # x_s.columns = data_p.columns
    # x_s, x_s_drop = train_test_split(x_s, train_size=0.5, random_state=np.random.randint(int(time.time())))
    # data = data.append(x_s)
    print '缩短边界并SMOTE处理后的P类样本规模：', data[data.label == 'p'].shape
    ##扩充d类样本
    data_d = data[data.label == 'd']
    print 'd类样本的初始规模:', data_d.shape
    ##去掉靠近主要特征(5BF)的边缘的样本-----本次缩短比例为上次的0.8
    # data_d = data_d[(data_d.length < 8000) & (data_d.emp < 40000)  & (data_d.aver_length < 2080) & (data_d.Assignment < 144)]
    # x = data_d.iloc[:, 0:-1]
    # y = data_d.iloc[:, -1]
    # s = Smote(np.array(x), 100, 4)
    # x_s = s.over_sampling()
    # x_s = pd.DataFrame(x_s)
    # x_s['label'] = 'd'
    # x_s.columns = data_d.columns
    # x_s, x_s_drop = train_test_split(x_s, train_size=0.5, random_state=np.random.randint(int(time.time())))
    # data = data.append(x_s)
    print '缩短边界并SMOTE处理后的d类样本规模：', data[data.label == 'd'].shape
    print '全部样本集的类别分布：', data['label'].value_counts()# 至此获得了扩充后的新数据集data

    #以总样本为基础划分训练数据和测试数据
    data_train, data_test = train_test_split(data, train_size=0.7, random_state=np.random.randint(int(time.time())))
    print '训练集整体分布：', data_train.shape
    print data_train.iloc[:, -1].value_counts()

    #寻找最容易被误判的点，分析反馈
    # d2n = []
    # n2d = []
    # n2p = []
    # p2n = []
    # for h in range(200): # 3.3 * 5
    #     data_train, data_test = train_test_split(data, train_size=0.75, random_state=np.random.randint(int(time.time())))
    #     clf = RandomForestClassifier(criterion='entropy')
    #     clf.fit(data_train.iloc[:, 1:-1], pd.Categorical(data_train.iloc[:, -1]).codes)
    #     y_test = pd.Categorical(data_test.iloc[:, -1]).codes
    #     y_pred = clf.predict(data_test.iloc[:, 1:-1])
    #     for i in range(len(data_test)):
    #         if ((y_test[i] == 0) & (y_pred[i] == 1)):
    #             d2n.append(int(data_test.iloc[i, 0]))
    #         if ((y_test[i] == 1) & (y_pred[i] == 0)):
    #             n2d.append(int(data_test.iloc[i, 0]))
    #         if ((y_test[i] == 1) & (y_pred[i] == 2)):
    #             n2p.append(int(data_test.iloc[i, 0]))
    #         if ((y_test[i] == 2) & (y_pred[i] == 1)):
    #             p2n.append(int(data_test.iloc[i, 0]))
    #         print '已完成', h, '轮的', i * 1.0 / (len(data_test)) * 100, '%'
    # joblib.dump(d2n, './model/d2n.list')
    # joblib.dump(n2d, './model/n2d.list')
    # joblib.dump(n2p, './model/n2p.list')
    # joblib.dump(p2n, './model/p2n.list')


    # 构建分类器
    ##构建单个分类器，测试效果（DT，SVM，BP）,借助特殊化的Bagging包
    #     # 随机降采样构造训练用子数据集(采样正负例各取约0.9， 0.1)
    x_train, y_train = GetDataSub(data_train, 0.25, 0.99)
    # clf = BaggingClassifier(DecisionTreeClassifier(criterion='entropy'), n_estimators=1, max_samples=1.0)
    # alpha = np.linspace(0.1, 5.5, 10)
    # clf = RandomizedSearchCV(SVC(kernel='rbf', class_weight={0: 3, 1: 1, 2: 1.6}), param_distributions={'C': alpha})
    # clf = RandomizedSearchCV(SVC(kernel='rbf'), param_distributions={'C': alpha})
    # clf = DecisionTreeClassifier(criterion='entropy')
    # clf = SVC(C=1.65, kernel='rbf')
    clf = RandomForestClassifier(criterion='entropy')
    # clf = KNeighborsClassifier()
    # clf = BaggingClassifier(RandomForestClassifier(criterion='entropy'), n_estimators=10)
    clf.fit(x_train, y_train)
    print '已用时：', time.time() - t
    # print clf.best_params_
    y_pred = clf.predict(data_test.iloc[:, 1:-1])
    y_test = pd.Categorical(data_test.iloc[:, -1]).codes
    print confusion_matrix(y_test, y_pred)
    print classification_report(y_test, y_pred)
    # joblib.dump(clf, './model/Bweak_single_debug8296.model')
    print '共用时：', time.time() - t



    #构建并训练分类器组（DT）
    # clf_Ls = []
    # n_clf = 5    #根据规模平均三轮遍历一整遍data_train，设定约遍历20遍
    # for i in range(n_clf):
    #     # 随机降采样构造训练用子数据集(采样正负例各取约0.9， 0.1)
    #     x_train, y_train = GetDataSub(data, 0.75, 0.99)
    #     # clf = DecisionTreeClassifier(criterion='entropy')
    #     # clf = SVC(C=1.65, kernel='rbf')
    #     # clf = RandomForestClassifier()
    #     # clf = AdaBoostClassifier()
    #     clf = KNeighborsClassifier()
    #     clf.fit(x_train, y_train)
    #     clf_Ls.append(clf)
    #     print '第', i, '个分类器训练完成'
    # joblib.dump(clf_Ls, './model/clf_Ls_RF_5_sub86.model')
    # print '全部训练完成，用时：', time.time() - t

    #缩小测试集，以缩短观察时间
    # data_test_noUse, data_test = train_test_split(data_test, train_size=0.65, random_state=np.random.randint(int(time.time())))

    # # 利用分类器组进行预测，并评价
    # print '测试集类别分布：', data_test.iloc[:, -1].value_counts()
    # clf_Ls = joblib.load('./model/clf_Ls_DecisionTree_99.model')
    # y_test = pd.Categorical(data_test.iloc[:, -1]).codes
    # sample_pred = []
    # sample_TotalCount = []
    # for i in range(len(data_test)):
    #     sample_preds = []
    #     sample_subCount = []
    #     for j in range(len(clf_Ls)):
    #         clf = clf_Ls[j]
    #         sample_preds.append(int(clf.predict(data_test.iloc[i, 0:-1])))
    #     sample_subCount = Sample_SubCount(sample_subCount, sample_preds)
    #     whoWin = VoteCompute(sample_preds)
    #     sample_TotalCount.append(sample_subCount)
    #     sample_pred.append(int(whoWin))
    #     print '计算第', i, '个样本..已完成', 100*(i*1.0/len(data_test)), '%'
    # print confusion_matrix(y_test, sample_pred)
    # print classification_report(y_test, sample_pred)
    # joblib.dump(sample_TotalCount, './model/clf_Ls_DecisionTree_99_resultAnalysis.list') #保存所有样本的评判结果以做反馈分析
    # print '共用时：', time.time() - t




# 得到结果后即收集误判样本以判断其大概的位置分布
def CountFallibilitySample(y_pred, y_test, data_test):
    nToD = []
    nToP = []
    dToN = []
    pToN = []
    for i in range(len(y_pred)):
        if ((y_test[i] == 1) & (y_pred[i] == 0)):
            nToD.append(data_test.iloc[i, :])
        if ((y_test[i] == 1) & (y_pred[i] == 2)):
            nToP.append(data_test.iloc[i, :])
        if ((y_test[i] == 0) & (y_pred[i] == 1)):
            dToN.append(data_test.iloc[i, :])
        if ((y_test[i] == 2) & (y_pred[i] == 1)):
            pToN.append(data_test.iloc[i, :])
    nToD = pd.DataFrame(nToD)
    nToP = pd.DataFrame(nToP)
    dToN = pd.DataFrame(dToN)
    pToN = pd.DataFrame(pToN)
    nToDP = nToD.append(nToP)
    pdToN = dToN.append(pToN)
    return nToDP.append(pdToN)


#统计单个样本的评判结果
def Sample_SubCount(sample_subCount, sample_preds):
    sample_subCount.append(sample_preds.count(0))
    sample_subCount.append(sample_preds.count(1))
    sample_subCount.append(sample_preds.count(2))
    return sample_subCount

#投票机制1：估测各类在单模型相同参数下预测时的平均精确值，综合各类样本
#         在测试集中所在的数量比例，最后乘以分类器（评审员）的数量即为
#         期望值。用实际出现的次数减去这个期望值再比上自身期望值所得
#         的这个公平的等比例差值，该值即为最终评判标准
#投票机制2（校正）：每一个样本的出现次数--每个样本被正判的平均概率*分类器（评审员）
#                的个数。此处的平均概率为单一分类器多次分类的估测结果。
#                小优化--正判概率*0.8
#投票机制3(更正) ：正判率为50%以上时，确实为得票数量最多的类型就是样本所属的类型
def VoteCompute(sample_preds):
    whoWin = Counter(sample_preds).most_common()[0][0]
    return whoWin
    # n_clf = len(sample_preds)
    # whoWin = 9 #默认分类类别为9，但一旦结果中出现9这个类则意味程序出错
    # if sample_preds.count(0) >= n_clf * 0.505:
    #     whoWin = 0
    # if sample_preds.count(1) >= n_clf * 0.9800:
    #     whoWin = 1
    # if sample_preds.count(2) >= n_clf * 0.7100:
    #     whoWin = 2
    # return whoWin


#降采样按一定比例随机构造训练子数据集(u，v为正负例采样率（初始为0.1， 0.9)）
def GetDataSub(data_train, u, v, h):
    # data = RepttitionSampling(data[data.label == 'n'], 0.1).append(RepttitionSampling(((data[data.label == 'p'])|(data[data.label == 'd'])), 0.9))
    data_n = data_train[data_train.label == 'n']
    data_pd = data_train[(data_train.label == 'p') | (data_train.label == 'd')]
    # data_n = RandomSampling(data_n, 0.36)
    data_n, data_n_noUse = train_test_split(data_n, train_size=u, random_state=int(time.time())/300000*h)#np.random.randint(int(time.time()/2))
    # data_pd = RandomSampling(data_pd, 0.95)
    data_pd, data_pd_noUse = train_test_split(data_pd, train_size=v, random_state=int(time.time())/2000*h)
    data_sub = data_pd.append(data_n)
    x_train = data_sub.iloc[:, 0:-1]
    y_train = pd.Categorical(data_sub.iloc[:, -1]).codes
    print '本次训练子集规模：', data_sub.iloc[:, -1].value_counts()
    return x_train, y_train


def RandomSampling(data, k):
    data_DeSamp = []
    for i in range(len(data)):
        if random.random() < k:
            data_DeSamp.append(data.iloc[i, :])
    return pd.DataFrame(data_DeSamp)

if __name__ == "__main__":
    t = time.time()
    warnings.filterwarnings('ignore')
    np.set_printoptions(threshold='nan', linewidth=800, suppress=True)
    pd.set_option('display.width', 800, 'max_rows', 10000, 'max_columns', 100)
    run()

