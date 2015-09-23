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


##############################################################

print('Loading train set...')
df_train= pd.read_csv("./input/train.csv",delimiter=',',header=0,nrows=15000)
#print('Loading test set...')
#df_test = pd.read_csv("./input/test.csv",delimiter=',',header=0)
print('Writing small subset...')
df_train.to_csv("./input/train_small.csv",index=False)