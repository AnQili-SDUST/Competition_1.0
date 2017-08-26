# -*- encoding: utf-8 -*-
"""
    Author:
    Name:
    Describe:
"""

import logging

logging.basicConfig()

# from sknn.mlp import Classifier, Layer
from sklearn import cross_validation
from sklearn import datasets
from sknn.mlp import Classifier, Layer
import sys

import time
import numpy as np

if len(sys.argv) == 1:
    print("ERROR: Please specify implementation to benchmark, 'sknn' or 'nolearn'.")
    sys.exit(-1)

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

from sklearn.base import clone
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report


mnist = fetch_mldata('mnist-original')
X_train, X_test, y_train, y_test = train_test_split(
        (mnist.data / 255.0).astype(np.float32),
        mnist.target.astype(np.int32),
        test_size=1.0/7.0, random_state=1234)


classifiers = []

if 'sknn' in sys.argv:
    from sknn.platform import gpu32
    from sknn.mlp import Classifier, Layer, Convolution

    clf = Classifier(
        layers=[
            # Convolution("Rectifier", channels=10, pool_shape=(2,2), kernel_shape=(3, 3)),
            Layer('Rectifier', units=200),
            Layer('Softmax')],
        learning_rate=0.01,
        learning_rule='nesterov',
        learning_momentum=0.9,
        batch_size=300,
        valid_size=0.0,
        n_stable=10,
        n_iter=10,
        verbose=True)
    classifiers.append(('sknn.mlp', clf))
    if 'nolearn' in sys.argv:
        from sknn.platform import gpu32
        from nolearn.lasagne import NeuralNet, BatchIterator
        from lasagne.layers import InputLayer, DenseLayer
        from lasagne.nonlinearities import softmax
        from lasagne.updates import nesterov_momentum

        clf = NeuralNet(
            layers=[
                ('input', InputLayer),
                ('hidden1', DenseLayer),
                ('output', DenseLayer),
            ],
            input_shape=(None, 784),
            output_num_units=10,
            output_nonlinearity=softmax,
            eval_size=0.0,

            more_params=dict(
                hidden1_num_units=200,
            ),

            update=nesterov_momentum,
            update_learning_rate=0.02,
            update_momentum=0.9,
            batch_iterator_train=BatchIterator(batch_size=300),

            max_epochs=10,
            verbose=1)
        classifiers.append(('nolearn.lasagne', clf))

        RUNS = 10

        for name, orig in classifiers:
            times = []
            accuracies = []
            for i in range(RUNS):
                start = time.time()

                clf = clone(orig)
                clf.random_state = int(time.time())
                clf.fit(X_train, y_train)

                accuracies.append(clf.score(X_test, y_test))
                times.append(time.time() - start)

            a_t = np.array(times)
            a_s = np.array(accuracies)

            y_pred = clf.predict(X_test)

            print("\n" + name)
            print("\tAccuracy: %5.2f%% ±%4.2f" % (100.0 * a_s.mean(), 100.0 * a_s.std()))
            print("\tTimes:    %5.2fs ±%4.2f" % (a_t.mean(), a_t.std()))
            print("\tReport:")
            print(classification_report(y_test, y_pred))







            # 手写的NN
# from NeuralNetwork import NeuralNetwork
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.metrics import confusion_matrix, classification_report
# import  warnings
# def run():
#     data = pd.read_csv('./data/train.csv', header=0)
#     print data.shape
#     data.dropna(inplace=True)
#     print data.shape
#     data['label'] = data.flag
#     data.__delitem__('flag')
#     x = np.array(data.iloc[:, 0:-1])
#     y = data.iloc[:, -1]
#     x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
#
#     y_train = LabelBinarizer().fit_transform(y_train)
#     y_test = pd.Categorical(y_test).codes
#     clf = NeuralNetwork([35, 1000, 3], 'logistic')
#     clf.fit(x_train, y_train, learning_rate=0.001, epochs=100000)
#     predictions = []
#     for i in range(len(x_test)):
#         o = clf.predict(x_test[i])
#         predictions.append(np.argmax(o))
#     print confusion_matrix(predictions, y_test)
#     print classification_report(predictions, y_test)
#
# if __name__ == '__main__':
#     warnings.filterwarnings('ignore')
#     run()


###pybrain
# from pybrain.supervised.trainers import BackpropTrainer
# from pybrain.tools.shortcuts import buildNetwork
# from pybrain.structure import TanhLayer, SigmoidLayer, SoftmaxLayer, LinearLayer
# from pybrain.supervised.trainers import BackpropTrainer
# from pybrain.datasets import SupervisedDataSet
# import numpy as np
#
#
# net = buildNetwork(2, 1, 1)
# net = buildNetwork(2, 1, 1, bias=True, hiddenclass=SigmoidLayer, outclass=LinearLayer)
# # print net['hidden0']
# # print net['in']
# # print net['out']
# # print net['bias']
# ds = SupervisedDataSet(2, 1)
# ds.addSample((0, 0), (0,))
# ds.addSample((0, 1), (1,))
# print net.activate([0, 0])
# # print type(np.array(ds))
# trainer = BackpropTrainer(net, ds)
# print trainer.trainUntilConvergence()