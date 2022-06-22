# -*- coding:utf-8 -*-
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

# 评价函数
def metric(label, predict):
    label_list = label.tolist()
    predict_list = predict.tolist()
    label_set = sorted(list(set(label_list + predict_list)))
    accuracy = np.mean(np.array(label_list) == np.array(predict_list))
    print("准确率为：%.2f%%" % (accuracy * 100))
    # 对每一个label求F1值
    TP_list, FP_list, FN_list = [], [], []
    for i in range(len(label_set)):
        TP, FP, FN = 0, 0, 0
        for j in range(len(label_list)):
            # 每次都以label_set[i]为正样本，其他的为负样本
            if label_list[j] == label_set[i]:
                if label_list[j] == predict_list[j]:
                    TP += 1
                else:
                    FN += 1
            else:
                if predict_list[j] == label_set[i]:
                    FP += 1
        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)

        if TP == 0:
            print("label_" + str(label_set[i]) + "的精确率为：%.3f，召回率为：%.3f，F1值为：%.3f" % (0, 0, 0))
            continue
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall)
        print("label_" + str(label_set[i]) + "的精确率为：%.3f，召回率为：%.3f，F1值为：%.3f" % (precision, recall, f1))
    precision_micro = sum(TP_list) / (sum(TP_list) + sum(FP_list))
    recall_micro = sum(TP_list) / (sum(TP_list) + sum(FN_list))
    if precision_micro == 0.0 and recall_micro == 0.0:
        print("微平均的精确率为：%.3f，召回率为：%.3f，F1值为：%.3f" % (0.000, 0.000, 0.000))
    else:
        f1_micro = (2 * precision_micro * recall_micro) / (precision_micro + recall_micro)
        print("微平均的精确率为：%.3f，召回率为：%.3f，F1值为：%.3f" % (precision_micro, recall_micro, f1_micro))

def average_value(list):
    return float(sum(list))/len(list)

def label_sum(label_train):
    label_sum=0
    for each in label_train:
        label_sum=label_sum+each
    return label_sum

# 对于不平衡数据进行采样
def generate_imbalance_data(X_train, y_train, imbalance):
    # 判断是否需要做不平衡处理
    # 如果训练集的正样本比例超过了40%，就不做不平衡处理
    if (label_sum(y_train) > (int(len(y_train) * 0.4))):
        print("The training data does not need balance.")
        return X_train, y_train
    else:
        # 对训练集中不平衡数据进行处理
        if imbalance == 'SMOTE':
            X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
        elif imbalance == 'BorderlineSMOTE':
            X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X_train, y_train)
        elif imbalance == 'ADASYN':
            X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)
        elif imbalance == 'SMOTEENN':
            X_resampled, y_resampled = SMOTEENN().fit_resample(X_train, y_train)
        elif imbalance == 'SMOTETomek':
            X_resampled, y_resampled = SMOTETomek().fit_resample(X_train, y_train)
        # 负采样
        else:
            rus = RandomUnderSampler(random_state=0, replacement=True)
            X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    # 打乱数据和标签
    state = np.random.get_state()
    np.random.shuffle(X_resampled)
    np.random.set_state(state)
    np.random.shuffle(y_resampled)

    return X_resampled, y_resampled
