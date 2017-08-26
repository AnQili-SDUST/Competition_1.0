# -*- encoding: utf-8 -*-
"""
    Author:
"""
import pandas as pd
import numpy as np
import warnings
from Competition.test_demo.ResultAnalysis_testA import Traversal_Count, ResultFromResult
import warnings

warnings.filterwarnings('ignore')
data_count = pd.DataFrame(columns={'id', 'n', 'p', 'd'})
for i in range(9):
    res = 'alpha_%d ' % (i)
    res = pd.read_csv('../result/BGsResulst_alpha_%d.csv'%(i))
    res.columns = ['ID', 'FLAG']
    print res.shape
    data_count = Traversal_Count(data_count, res, 1)
    print data_count.shape

alpha = pd.read_csv('../result/BGsResulst_alpha.csv', header=0)
data_count = Traversal_Count(data_count, res, 2)
svm = pd.read_csv('../result/svc_pure_4w_predict.csv', header=0)
data_count = Traversal_Count(data_count, res, 2)

data_count.to_csv('../result/ResultFromResult_BGs_alpha.csv', index=False)
ResultFromResult()

