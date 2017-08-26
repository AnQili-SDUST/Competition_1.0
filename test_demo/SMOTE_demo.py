# -*-  encoding: utf-8 -*-
from Competition.SMOTE import Smote
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,f_classif,chi2


np.set_printoptions(threshold='nan', linewidth=800, suppress=True)
pd.set_option('display.width', 800, 'max_rows', 10000, 'max_columns', 100)
##准备数据，并用smote生成采样样本
#
# a=np.array([[1,2,3],[4,5,6],[7,8,9],[0.5,1.5,2.5],[3.5,4.5,5.5],[6.5,7.5,8.5]])
# s=Smote(a, N=100, k=5)
# b = s.over_sampling()
# print b
#
# a = np.random.randint(0, 10, [10,10], dtype=int )
# s=Smote(a, N=100, k=5)
# b = s.over_sampling()
# a = pd.DataFrame(a)
# a['y'] = 0
# b =pd.DataFrame(b)
# b['y'] = 1
# data_all = b.append(a)
# x = np.array(data_all.iloc[:, 0:-1])
# y = np.array(data_all.iloc[:, -1])

#
# data = pd.read_csv('../data/train_1.csv', header=0)
# data.dropna(inplace=True)
# data['label'] = data['flag']
# data.__delitem__('flag')
# ##得到最好的两个特征，用此来作图
# x = data.iloc[:, 0:-1]
# y = data.iloc[:, -1]
# fs = SelectKBest(chi2, 2)
# fs.fit(x, y)
# idx = fs.get_support(True)
# print 'fs.get_support() = ', idx
# print data.__getitem__(idx)
# ##
# s = Smote(np.array(x), 100, 4)
# s.over_sampling()

# data = data[data.label == 'b']
# a = data[idx]
# a = np.array(a)
# s = Smote(a, N=300, k=5)
# b = pd.DataFrame(s.over_sampling())
# b['label'] = 1
# a = pd.DataFrame(a)
# a['label'] = 0
# data_all = b.append(a)
# x = np.array(data.iloc[:, 0:-1])
# y = np.array(data.iloc[:, -1])
#




#以data_d为例做smote,并选择2BestFeature作图
data = pd.read_csv('../data/train_1.csv', header=0)
data.dropna(inplace=True)
data['label'] = data['flag']
data.__delitem__('flag')
data_d = data[data.label == 'd']
x_d = data_d.iloc[:, 0:-1]
y_d = data_d.iloc[:, -1]
x_d = np.array(x_d)
s = Smote(x_d, 100, 5)
x_d_s = s.over_sampling()
x_d_s = pd.DataFrame(x_d_s)
x_d_s['label'] = 1
x_d_s.columns = data_d.columns
x_d = pd.DataFrame(x_d)
x_d['label'] = 0
x_d.columns = data_d.columns
x = x_d_s.append(x_d)
y = x.iloc[:, -1]

#选择2——BestFeature作图
fs = SelectKBest(chi2, 5)
fs.fit(data.iloc[:, 0:-1], data.iloc[:, -1])
idx = fs.get_support(indices=True)
x_label, y_label, z_label, u_label, v_label = data[idx].columns
# x_label, y_label, z_label = data[idx].columns
print data[idx].columns
x = x[idx]
print type(x)
x = x.values
print type(x)

#画图, 画出生成前后的样本图(红色为插值生成的样本)
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080'])
cm_dark = mpl.colors.ListedColormap(['g', 'r'])
mpl.rcParams['font.sans-serif'] = u'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
plt.figure(facecolor='w')
plt.scatter(x[:, 0], x[:, 1], s=5, c=y, marker='o', cmap=cm_dark)
plt.grid(b=True, ls=':')
plt.xlabel(x_label, fontsize=24)
plt.ylabel(y_label, fontsize=24)
plt.show()