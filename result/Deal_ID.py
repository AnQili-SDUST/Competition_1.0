# -*- encoding:utf-8 -*-
import pandas as pd
import warnings
import numpy as np

warnings.filterwarnings("ignore")
data = pd.read_csv('./Fill_ID/ResultFromResult.csv', header=0)
print data.shape
#添0
data_1 = data[data.ID < 100000]
data_2 = data[data.ID >= 100000]
for i in range(len(data_1)):
    data_1.iloc[i, 0] = str(data_1.iloc[i, 0]).zfill(6)
    print '处理至第', i, '行'
data_final = data_1.append(data_2)
data_final.to_csv('./Fill_ID/ResultFromRes.csv', index=False)
print data_final.shape