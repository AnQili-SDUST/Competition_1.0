# -*-encoding:utf-8 -*-
import numpy as np
import pandas as pd
from  collections import Counter
from sklearn.externals import joblib
ls = joblib.load('../model/clf_Ls_DecisionTree_99.list')
print len(ls)
ls = pd.DataFrame(ls)
ls_0 = ls[ls.iloc[:, 0] > 0]
print ls_0.shape
ls_0 = list(ls_0.iloc[:, 0])
ls_0.sort()
ls_0.reverse()
print ls_0
print ls_0[0:35]
print ls_0[35:62]
print ls_0[62:]
# print ls_0[:, 0]
# print type(ls)
# l =  pd.DataFrame(ls)
# print Counter(l.iloc[:, 0] != 0)
# l_0 = l.iloc[:, 0]
# l_0 = l_0[l_0 > 0]
# l_0 = list(l_0)
# l_0.sort()
# print l_0
# l_0.reverse()
# print l_0
# l_0_t = l_0[0:84]
# l_0_f = l_0[84: 160]
# l_0_noU = l_0[160:]
# print l_0_t
# print l_0_f
# print l_0_noU

