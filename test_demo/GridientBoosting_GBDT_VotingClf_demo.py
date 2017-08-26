






























# import  random
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# data = pd.read_csv('../data/train.csv', header=0)
# print data.shape
# print data.shape[1]
# print data['flag'].value_counts()
# data_sub = []
# for i in range(len(data)):
#     if random.random() <= 0.3:
#         data_sub.append(data.iloc[i, :])
# data_sub = pd.DataFrame(data_sub)
# print data_sub.shape
# print data_sub['flag'].value_counts()
#
# data_sub_large, data_sub_small = train_test_split(data, train_size=0.7, random_state=1)
# print data_sub_small.shape
# print data_sub_small['flag'].value_counts()
#
