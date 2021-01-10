#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys, os, platform
from tqdm import tqdm
import warnings
import argparse

#sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, f1_score, recall_score, precision_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from imblearn.over_sampling import RandomOverSampler

#matplotlib
import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.ticker as ticker

workdir = ''# change to working directory
    
def perf_measure(y_actual, y_hat):
    tp = fp = tn = fn = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1: tp += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]: fp += 1
        if y_actual[i]==y_hat[i]==0: tn += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]: fn += 1
    if (tp+fn) == 0: sensitivity = np.nan
    else: sensitivity = tp/(tp+fn) # recall
    if (tn+fp) == 0: specificity = np.nan
    else: specificity = tn/(tn+fp)
    if (tp+fp) == 0: ppv = np.nan
    else: ppv = tp/(tp+fp) # precision or positive predictive value (PPV)
    if (tn+fn) == 0: npv = np.nan
    else: npv = tn/(tn+fn) # negative predictive value (NPV)
    if (tp+tn+fp+fn) == 0: hitrate = np.nan
    else: hitrate = (tp+tn)/(tp+tn+fp+fn) # accuracy (ACC)
    return sensitivity, specificity, ppv, npv, hitrate # (tp, fp, tn, fn)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='DT', help='LR_L1, LR_L2, Adaboost, DT')
    parser.add_argument('--label', default='Suspected cases', help='Suspected cases, Infection cases')
    parser.add_argument('--upsampling', default=True, action='store_true')
    parser.add_argument('--test_ratio', default=0.2, help='Held-out testing set')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # =============================================================================
    # Read data
    # =============================================================================
    df = pd.read_excel(workdir + 'Developement_datasets_final.xls', sheet_name = '疑似')
    y = df[args.label]
    y_sus = df['Suspected cases']
    X = df[[c for c in df.columns if c not in ['Suspected cases', 'Infection cases']]]
    colnames = X.columns.astype(str)
    X = X.values
    # =============================================================================
    # Cross validation
    # =============================================================================
    np.random.seed(0)
    
    test_ratio = args.test_ratio
    rand_test_idx = np.arange(len(X))
    np.random.shuffle(rand_test_idx)
    rand_test_idx = rand_test_idx[0:int(test_ratio * len(X))]
    rand_train_val_idx = [i for i in np.arange(len(X)) if i not in rand_test_idx]
    
    X_test, y_test = X[rand_test_idx], y.values[rand_test_idx]
    y_sus_test = y_sus.values[rand_test_idx]
    X_trainval, y_trainval = X[rand_train_val_idx], y.values[rand_train_val_idx]

    print('standardize input.')
    stdscaler = StandardScaler().fit(X_trainval)
    X_trainval = stdscaler.transform(X_trainval)
    X_test = stdscaler.transform(X_test)
    stdscaler_df = pd.DataFrame(np.stack((stdscaler.mean_, stdscaler.scale_)).T, index = colnames, columns = ['mean_','scale_'])
    stdscaler_df.to_csv(workdir + 'stdscaler.csv')
    # =============================================================================
    #     Handling the imbalanced dataset
    # =============================================================================
    if args.upsampling:
        print('Handling the imbalanced dataset by RandomOverSampler ...')
        ros = RandomOverSampler(random_state=0)
        X_trainval, y_trainval = ros.fit_resample(X_trainval, y_trainval)
        X_trainval, y_trainval = shuffle(X_trainval, y_trainval, random_state=0)
    
#    class_weight = None
#    class_weight = 'balanced'
    class_weight = {0: 1/np.sum(y_trainval == 0), 1: 1/np.sum(y_trainval == 1)} # class_weightdict or ‘balanced’, default=None
    nfolds = 10
    
    if args.model == 'LR_L1':
        C_range = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])*0.25
    if args.model == 'LR_L2':
        C_range = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])*0.25
    if args.model == 'DT':
        C_range = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    elif args.model == 'Adaboost':
        C_range = [2**4, 2**5, 2**6, 2**7, 2**8]
    metrics = pd.DataFrame(index = C_range, columns = ['AUC', 'F1', 'Precision', 'Recall', 'sensitivity', 'specificity', 'ppv', 'npv', 'hitrate'], dtype=float)

    for C in C_range:
        AUC = f1 = precision = recall = 0
        sensitivity = specificity = ppv = npv = hitrate = 0
        for train_index, val_index in KFold(n_splits=nfolds, shuffle=True, random_state=10).split(X_trainval):
            X_train, y_train = X_trainval[train_index], y_trainval[train_index]
            X_val, y_val = X_trainval[val_index], y_trainval[val_index]
            if args.model == 'LR_L1':
                model = LogisticRegression(penalty='l1', C = C, solver = 'saga', random_state=0, class_weight = class_weight)
            if args.model == 'LR_L2':
                model = LogisticRegression(penalty='l2', C = C, solver = 'saga', random_state=0, class_weight = class_weight)
            if args.model == 'DT':
                model = DecisionTreeClassifier(max_depth = C, class_weight = class_weight)
            elif args.model == 'Adaboost':
                model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1, class_weight = class_weight), n_estimators = C)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
                y_val_pred, y_val_pred_proba = model.predict(X_val), model.predict_proba(X_val)[:,1]
                fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)
                AUC += auc(fpr, tpr)/nfolds
                f1 += f1_score(y_val_pred, y_val, average = 'binary')/nfolds
                precision += precision_score(y_val, y_val_pred, average = 'binary')/nfolds
                recall += recall_score(y_val, y_val_pred, average = 'binary')/nfolds
                sensitivity_tmp, specificity_tmp, ppv_tmp, npv_tmp, hitrate_tmp = perf_measure(y_val, y_val_pred)
                sensitivity += sensitivity_tmp/nfolds
                specificity += specificity_tmp/nfolds
                ppv += ppv_tmp/nfolds
                npv += npv_tmp/nfolds
                hitrate += hitrate_tmp/nfolds
                
        print('C: %f, AUC: %.4f, F1: %.4f, P: %.4f, R: %4f, Sen: %.4f, Spe: %.4f' % (C, AUC, f1, precision, recall, sensitivity, specificity))
        metrics.loc[C, :] = AUC, f1, precision, recall, sensitivity, specificity, ppv, npv, hitrate

    best_C = metrics.idxmax().F1
    print('Best penalty C: %.4f' % best_C)
    if args.model == 'LR_L1':
        model_fin = LogisticRegression(penalty='l1', C = best_C, solver = 'saga', random_state=0, class_weight = class_weight)
    if args.model == 'LR_L2':
        model_fin = LogisticRegression(penalty='l2', C = best_C, solver = 'saga', random_state=0, class_weight = class_weight)
    if args.model == 'DT':
        model_fin = DecisionTreeClassifier(max_depth = best_C, class_weight = class_weight)
    elif args.model == 'Adaboost':
        model_fin = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1, class_weight = class_weight), n_estimators = best_C)
    model_fin.fit(X_trainval, y_trainval)
    y_test_pred, y_test_pred_proba = model_fin.predict(X_test), model_fin.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
    AUC = auc(fpr, tpr)
    f1 = f1_score(y_test_pred, y_test, average = 'binary')
    precision = precision_score(y_test, y_test_pred, average = 'binary')
    recall = recall_score(y_test, y_test_pred, average = 'binary')
    sensitivity, specificity, ppv, npv, hitrate = perf_measure(y_test, y_test_pred)
    
    print('[Final report] Best C: %f, AUC: %.4f, F1: %.4f, P: %.4f, R: %4f, Sen: %.4f, Spe: %.4f' % (best_C, AUC, f1, precision, recall, sensitivity, specificity))

# =============================================================================
# Feature importance
# =============================================================================
    feature_name = [c for c in df.columns.values.astype(str) if c not in ['Suspected cases', 'Infection cases']]
    if args.model in ['LR_L1', 'LR_L2']:
        importance = model_fin.coef_.reshape(-1)
    elif args.model == 'DT':
        importance = model_fin.feature_importances_
    elif args.model == 'Adaboost':
        importance = model_fin.feature_importances_

    coefficients = pd.DataFrame(importance, index = feature_name, columns = ['Weight'])
    coefficients.sort_values(by='Weight', ascending=False, inplace=True)
    if args.model in ['LR_L1', 'LR_L2']:
        coefficients.loc['intercept_','Weight'] = model_fin.intercept_
    coefficients.to_csv(workdir + 'feature_importance_%s.csv' % args.model)

# =============================================================================
# External validation
# =============================================================================
    df_ev = pd.read_excel(workdir + 'validation_datasets_final.xls', sheet_name = '疑似', dtype = float)
    df_ev['Chills'] = 0
    y_ev = df_ev['Suspected cases']
    X_ev = df_ev[[c for c in df_ev.columns if c not in ['Suspected cases']]]
    X_ev = X_ev[colnames]
    colnames_ev = X_ev.columns.astype(str)
    X_ev = stdscaler.transform(X_ev)


    y_ev_pred, y_ev_pred_proba = model_fin.predict(X_ev), model_fin.predict_proba(X_ev)[:,1]
    fpr, tpr, thresholds = roc_curve(y_ev, y_ev_pred_proba)
    AUC = auc(fpr, tpr)
    f1 = f1_score(y_ev_pred, y_ev, average = 'binary')
    precision = precision_score(y_ev, y_ev_pred, average = 'binary')
    recall = recall_score(y_ev, y_ev_pred, average = 'binary')
    sensitivity, specificity, ppv, npv, hitrate = perf_measure(y_ev, y_ev_pred)
    
    print('[External Validation] Best C: %f, AUC: %.4f, F1: %.4f, P: %.4f, R: %4f, Sen: %.4f, Spe: %.4f' % (best_C, AUC, f1, precision, recall, sensitivity, specificity))







