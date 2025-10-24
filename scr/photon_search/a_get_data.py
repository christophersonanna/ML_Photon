# -*- coding: utf-8 -*-

#import awkward as ak
#import pyarrow.parquet as pq
import z_config as config
import pandas as pd #will change later

#loading data
#parquet_file = pq.ParquetFile(config.PROCESSED_DATA_PATH)
#parquet_file = pd.read_parquet(config.PROCESSED_DATA_PATH)
#print(parquet_file.columns)

#for testing the scripts for now
X_test = pd.read_csv(config.PROCESSED_DATA_PATH_TEST, delimiter=',')
X_train = pd.read_csv(config.PROCESSED_DATA_PATH_TRAIN, delimiter=',')


del(X_test['mir_id'])
del(X_test['mir_nmir'])
#del(X_test['mir_ngtube'])
del(X_test['Unnamed: 0'])

del(X_train['mir_id'])
del(X_train['mir_nmir'])
#del(X_train['mir_ngtube'])
del(X_train['Unnamed: 0'])

X_test = X_test
X_train = X_train

print(X_train.columns)