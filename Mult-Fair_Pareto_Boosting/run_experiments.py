# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:55:41 2020

@author: Arjun
"""
#from Averagor import AdaFair_Multi as averagor
from Maximus_optimized_new import Multi_Fair as maximus

from sklearn.model_selection import StratifiedShuffleSplit as ss
import numpy as np
from DataPreprocessing.my_utils import get_score,get_fairness,vis


def train_classifier1(X_train, X_test, y_train, y_test, sa_index, p_Group, base_learners,preference):

    classifier = maximus(n_estimators=base_learners, saIndex=sa_index, saValue=p_Group,preference=preference)

    classifier.fit(X_train, y_train)

    y_pred_probs = classifier.predict_proba(X_test)[:, 1]
    y_pred_labels = classifier.predict(X_test)
    #f=classifier.feature_importances_
    #return classifier.conf_scores, classifier.get_weights_over_iterations(), classifier.get_initial_weights()
    return y_pred_probs, y_pred_labels,classifier

            
def run_exp(dt='Bank',epochs=10):
    if dt=='Adult':
        from DataPreprocessing.load_adult import load_adult
        X, y, sa_index, p_Group, x_control,F = load_adult()
        preference=[0.4,0.4,0.2]
    elif dt=='Bank':
        from DataPreprocessing.load_bank import load_bank
        X, y, sa_index, p_Group, x_control,F = load_bank()
        preference=[0.4,0.4,0.2]
    elif dt=='Credit':
        from DataPreprocessing.load_credit import load_credit
        X, y, sa_index, p_Group, x_control,F = load_credit()
        preference=[0.33,0.34,0.33]
    elif dt=='Compas':
        from DataPreprocessing.load_compas_data import load_compas
        X, y, sa_index, p_Group, x_control,F = load_compas()
        preference=[0.33,0.34,0.33]
    elif dt=='KDD':
        from DataPreprocessing.load_kdd import load_kdd
        X, y, sa_index, p_Group, x_control,F = load_kdd()
        preference=[0,0.5,0.5]
        #sa_index=[sa_index[0],sa_index[3]] ##########for conducting the experiment with only Age and Sex uncomment this
    protected=[F[v] for v in sa_index]
    print(sa_index,protected)
    in_tr,in_ts,ytest=[],[],[]
    pred1,prob1,H_t=[],[],[]

    sss = ss(n_splits=epochs,test_size=0.4)
    base_learners=499

    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ytest.append(y_test)

        in_tr.append(train_index)
        in_ts.append(test_index)
        
        pb1,pd1,f1=train_classifier1(X_train, X_test, y_train, y_test, sa_index, p_Group,base_learners,preference)
        prob1.append(pb1)
        pred1.append(pd1)
        H_t.append(f1)
       
        

        

        
    if dt=='Compass' and 'age_cat' in protected:
        protected[0]='age'
    path='D:/L3S/Multi-Fair/Multi-versions/'   
    #np.save(path+dt+"preds_new.npy",np.array([pred1]))

    rs1=get_fairness(sa_index,p_Group,in_ts,pred1,X,y)
    pf1=get_score(pred1,in_ts,X,y)

    results=np.array([list(rs1.values())])
    performance=np.array([pf1])

    
    vis(path,results,performance,L=protected,dt=dt,clfs=['Multi-Fair'])
    
if __name__ == "__main__":
    run_exp(dt='Compas',epochs=10)       
     