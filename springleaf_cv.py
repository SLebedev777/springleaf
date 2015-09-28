# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:43:01 2015

@author: pups
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn import linear_model, cross_validation, preprocessing, tree
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.externals import joblib


##############################################################

print('Loading train set...')
good_features_list = joblib.load('./pickles/good_features_list.pkl')
good_features_list.append('target')
good_features_list.append('ID')
df_train = pd.read_csv("./input/train_40000_prepared.csv", delimiter=',', header=0, usecols=good_features_list)

df_target = df_train['target']
df_train = df_train.drop(['target'], axis=1)
print('...complete loading.')
print(df_train.shape)
BaseRate = np.count_nonzero(df_target.values)/len(df_target)

#####################################################
# DATA PREPROCESSING
#####################################################

train = df_train.values
target = df_target.values.ravel()


# заменить NaN на медиану в столбце
imp = preprocessing.Imputer(strategy='median')
train = imp.fit_transform(train)

train = preprocessing.scale(train)


Classifiers = {'LR': linear_model.LogisticRegression(C=0.001),
             #'NaiveBayes': GaussianNB(),          
             #'AdaBoost': AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=100,max_features='sqrt'),100),
             #'ExtraTrees': ExtraTreesClassifier(500,max_features='sqrt',max_depth=7,min_samples_leaf=10),
             #'RandomForest':RandomForestClassifier(100,max_features='sqrt',max_depth=None,
             #                                      min_samples_leaf=10,warm_start=False),
              }


###########################################################
# run tq cross-validation
##########################################################
plt.ioff()
plt.figure(2, (8, 6))
plt.xlim([-0.05, 1.])
plt.ylim([-0.05, 1.])
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
T, Q = 1, 5
print('Running TxQ cross-validation, T=', T, 'Q=', Q)
cfr_results = dict()
for cfr_name in Classifiers.keys():
    print('Training estimator: ', cfr_name)
    cfr = Classifiers[cfr_name]
    results = []
    cv_results = []
    i = 0
    for t in range(T):
        cv = cross_validation.StratifiedKFold(target, Q, shuffle=True)
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        acc = 0
        mean_acc = 0
        for traincv, testcv in cv:   # for Q loop
            probas = cfr.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
            acc = cfr.score(train[testcv], target[testcv])
            fpr, tpr, thresholds = roc_curve(target[testcv], probas[:, 1], pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            results.append(acc)   #,   [x[1] for x in probas] )
            mean_acc += acc
            print('learning, t=', t, 'q=', i, 'accuracy=', acc, 'AUC=', auc(fpr, tpr))
            i += 1
        mean_tpr /= Q
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        mean_acc /= Q
        cv_results.append((mean_fpr, mean_tpr, mean_auc, mean_acc, probas))

    cfr_mean_fpr = np.array([cv_results[i][0] for i in range(T)]).mean(0)
    cfr_mean_tpr = np.array([cv_results[i][1] for i in range(T)]).mean(0)
    cfr_mean_auc = np.mean([cv_results[i][2] for i in range(T)])
    cfr_mean_acc = np.mean([cv_results[i][3] for i in range(T)])
    plt.plot(cfr_mean_fpr, cfr_mean_tpr, '--',
             label=cfr_name+': acc=%0.2f' % cfr_mean_acc + ' AUC=%0.2f' % cfr_mean_auc, lw=2)
    cfr_results.setdefault(cfr_name, (cfr_mean_fpr, cfr_mean_tpr, cfr_mean_auc, cfr_mean_acc))
    print(cfr_name, 'mean AUC=', cfr_mean_auc, 'Mean accuracy=', cfr_mean_acc)
plt.legend(loc='lower right', fontsize='medium')
plt.show()



