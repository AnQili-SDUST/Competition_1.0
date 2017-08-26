# -*- encoding: utf-8 -*-
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import time
import random


print random.uniform(0.2, 0.3)
# d2n = joblib.load('../model/d2n.list')
# n2d = joblib.load('../model/n2d.list')
# n2p = joblib.load('../model/n2p.list')
# p2n = joblib.load('../model/p2n.list')
#
# print len(d2n)
# d2n_s = pd.Series(d2n)
# d2n_VCount = d2n_s.value_counts()
# d2n_s.drop_duplicates(inplace=True)
# print d2n_s.shape
# d2n_VCount.to_csv('../result/hardSamples/d2n_VCount.csv', index=True)
# d2n_s.to_csv('../result/hardSamples/d2n.csv', index=False)
#
# print len(n2d)
# n2d_s = pd.Series(n2d)
# n2d_s.drop_duplicates(inplace=True)
# print n2d_s.shape
# n2d_s.to_csv('../result/hardSamples/n2d.csv', index=False)
#
# print len(n2p)
# n2p_s = pd.Series(n2p)
# n2p_s.drop_duplicates(inplace=True)
# print n2p_s.shape
# n2p_s.to_csv('../result/hardSamples/n2p.csv', index=False)
#
# print len(p2n)
# p2n_s = pd.Series(p2n)
# p2n_s.drop_duplicates(inplace=True)
# print p2n_s.shape
# p2n_s.to_csv('../result/hardSamples/p2n.csv', index=False)



# temp = pd.read_csv('../data/temp.csv', header=0)
# print temp.shape
# print temp[temp.ID == 343467]
#
# ls = [1, 2, 3, 4, 5, 6, 7, 8, 3, 4,5,6,7,4,3,5,5,7]
# print type(ls.count(7))


# print np.zeros([5])
#
# res = pd.DataFrame([0,1,2])
# for i in range(5):
#     res['label%d' % (i)] = [9,8,7]
# print res


# lists = [1,2,2,2,2,0,1,2,0,1,1,1,0,0,0,0,0,2,2,2,2,1,0,2,1,0,2,1,0,0,2,1]
# print Counter(lists).most_common()[0][0]
# print type(Counter(lists).most_common()[0][0])

# for i in range(5):
#     print "He is %d years old"%(i)
#     print './result/BGsResulst_alpha_%d'%(i)
#
# ls = [2, 3, 1]
# print ls[2]
#
# emp = pd.DataFrame(columns=('a', 'b', 'c'))
# emp = emp.set_value(len(emp), 'b', 5)
# print emp
# emp = emp.set_value(len(emp)-1, 'c', 9)
# print emp
# new = pd.DataFrame({"name":"","age":"","sex":""},index=["0"])

# print np.zeros([3,4])
d = pd.DataFrame(columns={'id', 'd', 'p'})
d['id'] = [3, 4, 5, 4, 6, 8, 4]
d['d'] = [5, 6, 7, 2, 5, 7, 8]
d['p'] = [6, 7, 8, 4, 2, 2, 6]
print d
"""
    id d p
    3  5 6
    4  6 7
    5  7 8
"""
print Counter(d.iloc[1:-1, 2]).most_common()[0][0]
f = pd.DataFrame(columns={'p', 'd', 'id'})
f['p'] = ['e', 'f']
f['id'] = ['a', 'b']
f['d'] = ['c', 'd']
print f
print d.append(f)
print d[[1]]
d.drop(['p'], inplace=True)
print d
print d.iloc[:, [-2, -1]]
print d[1]
d_sub = d[d.d.isin([6])]
print d_sub
print d.loc[int(np.array(d_sub.index))].reshape(-1)
d.insert(0, 'ID', d.id)
d.drop(labels=['id'], axis=1, inplace=True)
print d
print d.loc[1, 'ID']
id = [4]
if(d[d.ID.isin(id)].empty):
    print 'kong'
else:
    print 'bukong'
print d
row = pd.DataFrame([(9, 8, 7)], columns={'ID', 'd', 'p'})
d = d.append(row, ignore_index=True)
d.set_value(len(d), 'ID', 12)
row = pd.DataFrame([(1, 2, 3)], columns={'ID', 'd', 'p'})
d = d.append(row, ignore_index=True)
print d

# d = pd.DataFrame
# s = pd.Series([4, 5, 6])
#
# s = pd.Series([9, 8, 7])
# d.append(s)
# s = pd.Series([5, 5, 5])
# d.append(s)
# s = pd.Series([9, 8, 7])
# d.append(s)
# print d


# clf_Ls = joblib.load('../model/clf_Ls_DT.model')
# print clf_Ls[1]
# print len(clf_Ls)

# print int(05), int(56.0)


# ls = [1, 2, 3, 4, 5, 6, 7, 8]
# a = pd.Series(ls, dtype=np.int32)
# print '列表', a.shape
# print '列表转矩阵', np.array(ls).shape
# print '列表转DataFrame',pd.DataFrame(ls).shape
# print '列表转矩阵转Series', pd.Series(np.array(ls).shape).shape
# print '列表转矩阵转DataFrame', pd.DataFrame(np.array(ls)).shape
# a.ravel()
#
# print ls.__len__()
# print a.shape



# y = np.array([2,3,5,6])
# d = pd.DataFrame()
# d['label'] = y
# d['id'] = y
# d['num'] = y+2
# d = d.iloc[:, [-3,-1]]
# print d