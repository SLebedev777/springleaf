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
from sklearn.feature_selection import RFECV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.externals import joblib
import gc

##############################################################
cfr_name='LR'
cfr=joblib.load(cfr_name+'.pkl')
print('Loaded the following trained estimator from Pickle:\n',cfr)

res=[]
ID=[]
df_test_chunks = pd.read_csv("./input/test_prepared.csv",delimiter=',',header=0,chunksize=40000)
for chunk in df_test_chunks:
    print('Reading and predicting chunk ID:',chunk['ID'].min(),' - ',chunk['ID'].max())    
    test=chunk.values
    imp=preprocessing.Imputer(strategy='median')
    test=imp.fit_transform(test)
    test=preprocessing.scale(test)
    res.extend(cfr.predict_proba(test)[:,1])
    ID.extend(chunk['ID'])
    

print('Writing results to file...')
out=pd.DataFrame(columns=['ID','Probability'])
out['ID']=ID
out['Probability']=res
out.to_csv('SL_springleaf_'+cfr_name+'.csv',sep=',',index=False)

del df_test_chunks,test,res,ID,out
gc.collect()


