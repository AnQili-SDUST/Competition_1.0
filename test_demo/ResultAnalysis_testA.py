# -*- encoding: utf-8 -*-
"""
    Author:
"""
import pandas as pd
import numpy as np
import warnings

def run():
    # #读进数据
    # data_bs = pd.read_csv('../result/Fill_ID/B_S63.csv', header=0)
    # data_bd = pd.read_csv('../result/Fill_ID/B_D99.csv', header=0)
    # data_cr = pd.read_csv('../result/Fill_ID/CR_CS13v1_SD99.csv', header=0)
    # data_count = pd.DataFrame(columns={'id', 'n', 'p', 'd'})
    # # data_count.insert(0, 'ID', data_count.id)
    # # data_count.drop(['id'], axis=1, inplace=True)
    # print data_bs.shape, data_bd.shape, data_cr.shape
    # data_count = Traversal_Count(data_count, data_bs)
    # data_count = Traversal_Count(data_count, data_bd)
    # data_count = Traversal_Count(data_count, data_cr)
    # data_count.to_csv('./ResultAnalysis_testA.csv', index=False)
    # print data_count.shape

    resFromResult = ResultFromResult()
    print resFromResult.shape
    resFromResult.to_csv('../result/ResultFromResult_BGs_alpha_res.csv', index=False)


def ResultFromResult():
    data = pd.read_csv('../result/ResultFromResult_BGs_alpha.csv', header=0)
    print data.shape
    #
    data_p = data[(data.p >= 2) & (data.d <=3)]
    data_p.__delitem__('p')
    data_p.__delitem__('d')
    data_p['FLAG'] = 'p'
    #
    data_d = data[data.d >= 1]
    data_d.__delitem__('p')
    data_d.__delitem__('d')
    data_d['FLAG'] = 'd'
    #
    data = data_d.append(data_p)
    data.__delitem__('n')
    return data
def Traversal_Count(data_count, data_x, k):
    for i in range(len(data_x)):
        data_x_in = data_count[data_count.id.isin([data_x.loc[i, 'ID']])]
        if (data_x_in.empty):
            if data_x.loc[i, 'FLAG'] == 'd':
                data_count.set_value(len(data_count), 'id', data_x.loc[i, 'ID'])
                data_count.set_value(len(data_count) - 1, 'd', 1*k)
                data_count.set_value(len(data_count) - 1, 'p', 0)
                data_count.set_value(len(data_count) - 1, 'n', 0)
            if data_x.loc[i, 'FLAG'] == 'n':
                data_count.set_value(len(data_count), 'id', data_x.loc[i, 'ID'])
                data_count.set_value(len(data_count) - 1, 'n', 1*k)
                data_count.set_value(len(data_count) - 1, 'd', 0)
                data_count.set_value(len(data_count) - 1, 'p', 0)
            if data_x.loc[i, 'FLAG'] == 'p':
                data_count.set_value(len(data_count), 'id', data_x.loc[i, 'ID'])
                data_count.set_value(len(data_count) - 1, 'p', 1*k)
                data_count.set_value(len(data_count) - 1, 'd', 0)
                data_count.set_value(len(data_count) - 1, 'n', 0)
        else:
            # print data_x_in.shape
            # print data_x_in
            editedRowNumber = int(np.array(data_x_in.index))
            if data_x.loc[i, 'FLAG'] == 'd':
                data_count.set_value(editedRowNumber, 'd', data_count.loc[editedRowNumber, 'd'] + 1*k)#其实有也且只有一行
            if data_x.loc[i, 'FLAG'] == 'n':
                data_count.set_value(editedRowNumber, 'n', data_count.loc[editedRowNumber, 'n'] + 1*k)
            if data_x.loc[i, 'FLAG'] == 'p':
                data_count.set_value(editedRowNumber, 'p', data_count.loc[editedRowNumber, 'p'] + 1*k)
    return data_count



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    run()