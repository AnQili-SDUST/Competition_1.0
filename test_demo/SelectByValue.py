import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier,BaggingClassifier, AdaBoostClassifier
from sklearn import XGBoost

GradientBoostingClassifier()




# 按某条件来选取部分dataframe
df = pd.DataFrame(np.random.randn(6, 4), index=list('123456'), columns=list('ABCD'))
df_ac = df[(df.A > 0) & (df.C > 0)]
print df
print df.drop([df.A>0])
