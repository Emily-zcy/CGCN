import numpy as np

def metric(label, predict):
    # label_list = label.tolist()
    # predict_list = predict.tolist()
    label_list = label
    predict_list = predict
    label_set = sorted(list(set(label_list + predict_list)))
    accuracy = np.mean(np.array(label_list) == np.array(predict_list))
    print("accuracy：%.2f%%" % (accuracy * 100))
    TP_list, FP_list, FN_list = [], [], []
    for i in range(len(label_set)):
        TP, FP, FN = 0, 0, 0
        for j in range(len(label_list)):
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
            print("label_" + str(label_set[i]) + "precision：%.3f，recall：%.3f，F1：%.3f" % (0, 0, 0))
            continue
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall)
        print("label_" + str(label_set[i]) + "precision：%.3f，recall：%.3f，F1：%.3f" % (precision, recall, f1))
    precision_micro = sum(TP_list) / (sum(TP_list) + sum(FP_list))
    recall_micro = sum(TP_list) / (sum(TP_list) + sum(FN_list))
    if precision_micro == 0.0 and recall_micro == 0.0:
        print("micro precision：%.3f，recall：%.3f，F1：%.3f" % (0.000, 0.000, 0.000))
    else:
        f1_micro = (2 * precision_micro * recall_micro) / (precision_micro + recall_micro)
        print("micro precision：%.3f，recall：%.3f，F1：%.3f" % (precision_micro, recall_micro, f1_micro))