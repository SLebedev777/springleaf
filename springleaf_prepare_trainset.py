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



#############################################################
# returns NumPy array of clean dataset
def PrepareDataset(ds):
  
  droplist=[]
  ds=ds.drop(['VAR_0340'],axis=1)
  for i in ds.columns:
      if np.dtype(ds[i])==np.dtype('O'): #добавить строковые столбцы в список на удаление
          droplist.append(i)
    
      else:
          #обработка No-Value
          colmax=ds[i].max()
          colmin=ds[i].min()
          print(i,'Min=',colmin,'Max=',colmax)
          if colmax>90 and colmax%9 == 0 and (colmax+1)%10 == 0: # если максимум - это одни девятки
              ds[i]=ds[i].replace(np.arange(colmax-9,colmax+1),np.nan)
          if colmin<0:                             # если минимум - отриц. число
              ds[i]=ds[i].replace(colmin,np.nan)               
              #ds[i]=ds[i]*(ds[i]<0) # все отриц. заменить нулями
      
  ds=ds.drop(droplist,axis=1) # удалить все ненужные столбцы        
    
  ds=ds.fillna(np.round(ds.median())) # fill NA
  
  #ds.describe().transpose().to_csv('train_small_prepared_stats.csv')
  return ds

##############################################################
def PlotAllHists(ds):
  for i in ds.columns:
      print('saving hist ',i)
      a=train[i].hist(bins=50)
      plt=a.get_figure()
      plt.savefig('hist_'+str(i)+'.png')
      plt.close()
##############################################################
      
      

print('Loading train set...')
df_train= pd.read_csv("./input/train.csv",delimiter=',',header=0,nrows=40000)
#print('Loading test set...')
#df_test = pd.read_csv("./input/test.csv",delimiter=',',header=0)
df_target = df_train['target']
#df_train=df_train.drop(['target'],axis=1)
print('...complete loading.')

BaseRate=np.count_nonzero(df_target.values)/len(df_target)

#####################################################
# DATA PREPROCESSING
#####################################################

train=PrepareDataset(df_train)
print('saving...')
train.to_csv("./input/train_40000_prepared.csv",delimiter=',',index=False)
'''
test=PrepareDataset(df_test)
target=df_target.values.ravel()


# заменить NaN на медиану в столбце
imp=preprocessing.Imputer(strategy='median')
train=imp.fit_transform(train)
test=imp.fit_transform(test)
'''
