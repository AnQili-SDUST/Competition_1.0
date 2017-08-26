
import  pandas as pd
a = pd.read_csv('./rfr.csv', header=0)
b = pd.read_csv('./2W_eighteen.csv', header=0)
total = a.append(b)
print a.shape, b.shape
# total.to_csv('./total_1.csv',index=False)
print len(total.ID.unique())