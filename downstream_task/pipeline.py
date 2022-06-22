# -*- coding:utf-8 -*-
import time
import tensorflow as tf
from downstream_task.utils import *
from sklearn.model_selection import RepeatedKFold
import numpy as np
from downstream_task.ClassifierOutput import *
import pandas as pd

# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

######################################################################################################################
baseURL = "./data/"
######################################################################################################################

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
# 'LogisticRegression', 'DecisionTree', 'RandomForest', 'MLP'
flags.DEFINE_string('classifier', 'MLP', 'Select a classifier for classification.')
# 'SMOTE', 'SMOTETomek', 'underSample'
flags.DEFINE_string('imbalance', 'underSample', 'Select a methods of dealing with imbalanced data.')

def run_evaluation(X_train, y_train, X_test, y_test):
    t = time.time()
    # data sample
    X_resampled, y_resampled = generate_imbalance_data(X_train, y_train, FLAGS.imbalance)

    # training classifier
    predprob_auc, predprob, precision, recall, fmeasure, auc, mcc, accuracy = \
        classifier_output(FLAGS.classifier, X_resampled, y_resampled, X_test, y_test,
                          grid_sear=True)  # False is only for debugging.

    print("precision=", "{:.5f}".format(precision),
          "recall=", "{:.5f}".format(recall),
          "f-measure=", "{:.5f}".format(fmeasure),
          "auc=", "{:.5f}".format(auc),
          "accuracy=", "{:.5f}".format(accuracy),
          "time=", "{:.5f}".format(time.time() - t))
    return precision, recall, fmeasure, auc, accuracy

# cross-version/cross-project: setting1
def load_train_test1(datalist):
    # loading embedding matrix and labels
    # The first version is used as training set
    origin_train_data = pd.read_csv(baseURL+datalist[0].split('-')[0]+"\\"+datalist[0]+"\\Process-Binary.csv", header=0, index_col=False)
    dw_train_data = pd.read_csv(baseURL+datalist[0].split('-')[0]+"\\"+datalist[0]+"\\CGCN_emb_add.csv", header=0, index_col=False)

    X_train = np.array(pd.concat([dw_train_data, origin_train_data.iloc[:, 3:-1]], axis=1))
    # X_train = np.array(origin_train_data.iloc[:, 3:-1])
    y_train = np.array(origin_train_data['bug'])

    # The second version is used as test set
    origin_test_data = pd.read_csv(baseURL + datalist[1].split('-')[0] + "\\" + datalist[1] + "\\Process-Binary.csv", header=0, index_col=False)
    dw_test_data = pd.read_csv(baseURL + datalist[1].split('-')[0] + "\\" + datalist[1] + "\\CGCN_emb_add.csv", header=0,
                               index_col=False)
    X_test = np.array(pd.concat([dw_test_data, origin_test_data.iloc[:, 3:-1]], axis=1))
    # X_test = np.array(origin_test_data.iloc[:, 3:-1])
    y_test = np.array(origin_test_data['bug'])
    return X_train, y_train, X_test, y_test

# cross-version/cross-project: setting2
def load_train_test2(datalist):
    # The last one is used as test set, the others as traing set
    # Training set
    for i in range(len(datalist) - 1):
        origin_train_data = pd.read_csv(baseURL + datalist[i].split('-')[0] + "\\" + datalist[i] + "\\Process-Binary.csv", header=0,
                                        index_col=False)
        dw_train_data = pd.read_csv(baseURL + datalist[i].split('-')[0] + "\\" + datalist[i] + "\\CGCN_emb_add.csv", header=0,
                                    index_col=False)
        if not i:
            X_train = np.array(pd.concat([dw_train_data, origin_train_data.iloc[:, 3:-1]], axis=1))
            # X_train = np.array(dw_train_data)
            y_train = np.array(origin_train_data['bug'])
        else:
            X_train_temp = np.array(pd.concat([dw_train_data, origin_train_data.iloc[:, 3:-1]], axis=1))
            # X_train_temp = np.array(dw_train_data)
            y_train_temp = np.array(origin_train_data['bug'])
            X_train = np.vstack((X_train, X_train_temp))
            y_train = np.hstack((y_train, y_train_temp))

    # Test set
    origin_test_data = pd.read_csv(baseURL + datalist[-1].split('-')[0] + "\\" + datalist[-1] + "\\Process-Binary.csv", header=0,
                                   index_col=False)
    dw_test_data = pd.read_csv(baseURL + datalist[-1].split('-')[0] + "\\" + datalist[-1] + "\\CGCN_emb_add.csv", header=0, index_col=False)

    X_test = np.array(pd.concat([dw_test_data, origin_test_data.iloc[:, 3:-1]], axis=1))
    # X_test = np.array(dw_test_data)
    y_test = np.array(origin_test_data['bug'])
    return X_train, y_train, X_test, y_test

# within-version
def load_within_train_test(data):
    F1_list = []
    precision_list = []
    recall_list = []
    AUC_list = []
    accuracy_list = []
    origin_train_data = pd.read_csv(baseURL + data.split('-')[0] + "\\" + data + "\\Process-Binary.csv", header=0,
                                    index_col=False)
    dw_train_data = pd.read_csv(baseURL + data.split('-')[0] + "\\" + data + "\\CGCN_emb_add.csv", header=0,
                                index_col=False)
    X = np.array(pd.concat([dw_train_data, origin_train_data.iloc[:, 3:-1]], axis=1))
    y = np.array(origin_train_data['bug'])
    exp_cursor = 1
    kf = RepeatedKFold(n_splits=5, n_repeats=5)  # We can modify n_repeats when debugging.
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        precision, recall, fmeasure, auc, accuracy = run_evaluation(X_train, y_train, X_test, y_test)
        F1_list.append(fmeasure)
        precision_list.append(precision)
        recall_list.append(recall)
        AUC_list.append(auc)
        accuracy_list.append(accuracy)

        exp_cursor = exp_cursor + 1

    avg = []
    avg.append(average_value(precision_list))
    avg.append(average_value(recall_list))
    avg.append(average_value(F1_list))
    avg.append(average_value(AUC_list))
    avg.append(average_value(accuracy_list))

    name = ['precision', 'recall', 'F1', 'AUC', 'Accuracy']
    results = []
    results.append(precision_list)
    results.append(recall_list)
    results.append(F1_list)
    results.append(AUC_list)
    results.append(accuracy_list)
    df = pd.DataFrame(data=results)
    df.index = name
    df.insert(0, 'avg', avg)
    df.to_csv(baseURL + "results\\within_project\\CGCN\\"+data+".csv")

# loop eight projects
dict_file=open('./within_project.txt','r')
lines=dict_file.readlines()
for line in lines:
    datalist = line.strip().split(',')
    print(line.strip() + " Start!")
    # dataset = datalist[0].split('-')[0]
    # print(datalist[-1] + " Start!")

    # within project
    load_within_train_test(line.strip())

    # cross version/cross project
    # X_train, y_train, X_test, y_test = load_train_test1(datalist)
    # precision,recall,fmeasure,auc,accuracy = run_evaluation(X_train, y_train, X_test, y_test)
    #
    # print(datalist[-1] + " Optimization Finished!")
    #
    # result = [line.strip(),precision,recall,fmeasure,auc,accuracy]
    # df = pd.DataFrame([result])
    # df.to_csv(baseURL+"results\\cross_project\\CGCN_cross.csv", mode='a', header=None,index=False)

# name = ['project', 'precision', 'recall', 'F1', 'AUC', 'accuracy']
# df = pd.read_csv(baseURL+"results\\cross_project\\CGCN_cross.csv",header=None,names=name)
# df.to_csv(baseURL+"results\\cross_project\\CGCN_cross.csv",index=False)