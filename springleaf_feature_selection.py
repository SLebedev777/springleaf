# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:43:01 2015

@author: pups
"""
import os as os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import interp
from sklearn import datasets,linear_model, cross_validation,svm,preprocessing,tree
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.feature_selection import *
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.externals import joblib


##############################################################

print('Loading train set...')
df_train= pd.read_csv("./input/train_40000_prepared.csv",delimiter=',',header=0)
#print('Loading test set...')
#df_test = pd.read_csv("./input/test.csv",delimiter=',',header=0)
df_target = df_train['target']
df_train=df_train.drop(['target'],axis=1)
print('...complete loading.')

BaseRate=np.count_nonzero(df_target.values)/len(df_target)

#####################################################
# DATA PREPROCESSING
#####################################################

train=df_train.values
target=df_target.values.ravel()


# заменить NaN на медиану в столбце
imp=preprocessing.Imputer(strategy='median')
train=imp.fit_transform(train)

#sel = VarianceThreshold(0.5)
#train=sel.fit_transform(train)

train=preprocessing.scale(train)



Classifiers={#'K-NN': KNeighborsClassifier(5),
             'SVM': svm.SVC(C=0.0025,kernel='linear', probability=True),
             'LR': linear_model.LogisticRegression(C=0.001),
             #'NaiveBayes': GaussianNB(),          
             #'AdaBoost': AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=100,max_features='sqrt'),100),
             #'ExtraTrees': ExtraTreesClassifier(500,max_features='sqrt',max_depth=7,min_samples_leaf=10),
             #'RandomForest':RandomForestClassifier(100,max_features='sqrt',max_depth=None,
             #                                      min_samples_leaf=10,warm_start=False),
             #'DecisionTree': tree.DecisionTreeClassifier(criterion='gini', 
             #                                            min_samples_split=20,max_depth=10,
             #                                            min_samples_leaf=10)
            }
             
cfr=Classifiers['LR']

'''
print('eigen values of data:')
eigvalues=np.linalg.eigvals(np.dot(train.T,train))
print(eigvalues)
print('running PCA')
pca = PCA(n_components=4)
pca.fit(train)
print('PCA variances of the features:', pca.explained_variance_)
'''


#sel = VarianceThreshold(0.5)
#train=sel.fit_transform(train)
#res=sel.get_support(True)
#print(res.shape)


#найти информативные признаки
#print('table of feature correlations:')
#print(np.corrcoef(train,rowvar=0))
cv=cross_validation.StratifiedKFold(target,5,shuffle=True)
rfecv = RFECV(cfr, step=5, cv=cv, scoring='roc_auc',verbose=2)
rfecv.fit(train, target)
print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure(3)
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

print('Writing good features list to Pickle...')
features_names=df_train.columns
good_features_bool=rfecv.get_support(False)
good_features_list=[i for i in list(map(lambda x,y: x and y,good_features_bool,features_names)) if i!=False]
joblib.dump(good_features_list,'good_features_list.pkl')



# стандартизация выборки признаков
#train=preprocessing.scale(train)
#test=preprocessing.scale(test)
#print('Training and test sets are standartized to mean=0 and std=1')

