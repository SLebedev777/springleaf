# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:43:01 2015

@author: pups
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
import gc

##############################################################
cfr_name = 'LR'
cfr = joblib.load('./pickles/' + cfr_name + '.pkl')
print('Loaded the following trained estimator from Pickle:\n', cfr)

res = []
ID = []
df_test_chunks = pd.read_csv("./input/test_prepared.csv", delimiter=',', header=0, chunksize=40000)
for chunk in df_test_chunks:
    print('Reading and predicting chunk ID:', chunk['ID'].min(), ' - ', chunk['ID'].max())
    test_set = chunk.values
    imp = preprocessing.Imputer(strategy='median')
    test_set = imp.fit_transform(test_set)
    test_set = preprocessing.scale(test_set)
    res.extend(cfr.predict_proba(test_set)[:, 1])
    ID.extend(chunk['ID'])
    
output_filename = './output/SL_springleaf_' + cfr_name + '.csv'
print('Writing results to file...', output_filename)
out = pd.DataFrame(columns=['ID', 'target'])
out['ID'] = ID
out['target'] = res
out.to_csv(output_filename, sep=',', index=False)

del df_test_chunks, test_set, res, ID, out
gc.collect()


