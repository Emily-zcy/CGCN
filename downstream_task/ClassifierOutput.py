# -*- coding:utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore",category=ConvergenceWarning)
    from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

def classifier_output(classifier_name,data_train,label_train,data_test,label_test,grid_sear=False):
    
    weight_dict={0:1, 1:2}
    if(classifier_name=="LogisticRegression"):
        rf = LogisticRegression(class_weight=weight_dict, solver='liblinear')
        if(grid_sear==False):
            rf.fit(data_train, label_train)
            predprob=rf.predict(data_test)
            predprob_auc=rf.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy
        if(grid_sear==True):
            parameters = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
            gsearch = GridSearchCV(rf, parameters, scoring='f1', cv=5, n_jobs=8)
            gsearch.fit(data_train, label_train)
            predprob=gsearch.predict(data_test)
            predprob_auc=gsearch.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy

    elif(classifier_name=="DecisionTree"):
        rf = DecisionTreeClassifier(class_weight=weight_dict)
        if(grid_sear==False):
            rf.fit(data_train, label_train)
            predprob=rf.predict(data_test)
            predprob_auc=rf.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy
        if(grid_sear==True):
            parameters = {'criterion':['gini','entropy'],'max_depth':[30,50,60,100],'min_samples_leaf':[2,3,5,10],'min_impurity_decrease':[0.1,0.2,0.5]}
            gsearch = GridSearchCV(rf, parameters, scoring='f1', cv=5, n_jobs=8)
            gsearch.fit(data_train, label_train)
            predprob=gsearch.predict(data_test)
            predprob_auc=gsearch.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy

    elif(classifier_name=="RandomForest"):
        rf = RandomForestClassifier(class_weight=weight_dict)
        if(grid_sear==False):
            rf.fit(data_train, label_train)
            predprob=rf.predict(data_test)
            predprob_auc=rf.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy
        if(grid_sear==True):
            print("Start Grid Search")
            # parameters = {'n_estimators':range(10,71,10), 'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20), 'min_samples_leaf':range(10,60,10)}
            parameters = {'n_estimators':range(10,71,10), 'max_depth':range(3,14,2), 'min_samples_split':range(10,201,20), 'min_samples_leaf':range(10,60,10)}
            gsearch = GridSearchCV(rf, parameters, scoring='f1', cv=3, n_jobs=14)
            gsearch.fit(data_train, label_train)
            predprob=gsearch.predict(data_test)
            predprob_auc=gsearch.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy
        
    elif(classifier_name=="MLP"):
        rf = MLPClassifier(random_state=10)
        if(grid_sear==False):
            rf.fit(data_train, label_train)
            predprob=rf.predict(data_test)
            predprob_auc=rf.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy
        if(grid_sear==True):
            parameters = {"hidden_layer_sizes": [(100,), (100, 30)], "solver": ['adam', 'sgd'], "max_iter": [200, 400],}
            gsearch = GridSearchCV(rf, parameters, scoring='f1', cv=5, n_jobs=8)
            gsearch.fit(data_train, label_train)
            predprob=gsearch.predict(data_test)
            predprob_auc=gsearch.predict_proba(data_test)[:, 1]
            recall=metrics.recall_score(label_test,predprob)
            auc=metrics.roc_auc_score(label_test,predprob_auc)
            precision=metrics.precision_score(label_test,predprob)
            fmeasure=metrics.f1_score(label_test,predprob)
            mcc=metrics.matthews_corrcoef(label_test,predprob)
            accuracy = metrics.accuracy_score(label_test, predprob)
            return predprob_auc,predprob,precision,recall,fmeasure,auc,mcc,accuracy
