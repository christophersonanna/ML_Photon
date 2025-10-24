#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 12:00:17 2025

@author: anna
"""
import z_config as config
import pandas as pd #will change later
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

#for testing the scripts for now
MC_data = pd.read_csv('/home/anna/photon_detection/Photon Machine Learning/data/gw_mc_all.data-combined(core_correction).csv', delimiter=',')
#toy_data = pd.read_csv('/home/anna/photon_detection/Photon Machine Learning/data/toy_data.csv', delimiter=',')
toy_data = pd.read_csv('/home/anna/photon_detection/Photon Machine Learning/data/toy_data_no_mir.csv', delimiter=',')
#X_test = pd.read_csv(config.PROCESSED_DATA_PATH_TEST, delimiter=',')
#X_train = pd.read_csv(config.PROCESSED_DATA_PATH_TRAIN, delimiter=',')
#MC_data = pd.concat([X_test,X_train], ignore_index=True)

del(MC_data['Unnamed: 0'])
del(MC_data['mir_id'])
del(MC_data['mir_nmir'])
del(MC_data['mir_ngtube'])


print(MC_data.columns)
print(toy_data.columns)
columns = ['xcore', 'ycore', 'th', 'phi', 'rp', 'psi', 'en', 'xf', 'xm', 'dxm',
       'c2t', 'c2p', 'xl', 'sz', 'fscin', 'fckov', 'fscat', 'mir_id',
       'mir_nmir', 'mir_ngtube', 'mcip']
    
print(columns[3])
#%%
plt.figure(figsize=(8,6))
plt.hist(MC_data['en'], bins=25, fill=False, density=True, edgecolor='r')
plt.hist(toy_data['en'], bins=25, fill=False, density=True, edgecolor='b')
plt.show()
#%% cross correlation
#individual
matrix = MC_data.corr()

plt.figure(figsize=(8,6))
sns.heatmap(matrix, annot=False, cmap="viridis", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of MC data")
plt.show()

matrix = toy_data.corr()

plt.figure(figsize=(8,6))
sns.heatmap(matrix, annot=False, cmap="viridis", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Toy data")
plt.show()
#%%means of values
print('MC:')
for i in range(0,len(columns)):
    print(columns[i])
    print(np.mean(MC_data[columns[i]]))

print('\nToy:')
for i in range(0,len(columns)):
    print(columns[i])
    print(np.mean(toy_data[columns[i]]))


#%%
#compare directly
corr = MC_data.corrwith(toy_data)
print(corr)

#%%histogram
for i in range(0,len(columns)):
    plt.hist(MC_data[columns[i]], histtype="stepfilled", fill=False, edgecolor='r', density=True, bins=19, label='MC Data')
    plt.hist(toy_data[columns[i]],histtype="stepfilled", fill=False, edgecolor='b', density=True, bins=19, label='Toy Data') 
    plt.title(f'{columns[i]}')
    plt.legend()
    plt.show()
    
#%%
print(max(MC_data['xm']))
print(max(toy_data['xm']))
MC_xmax01 = MC_data.nlargest(int(len(MC_data)*.01), 'xm')
Toy_xmax01 = MC_data.nlargest(int(len(MC_data)*.01), 'xm')

MC_xmax05 = MC_data.nlargest(int(len(MC_data)*.05), 'xm')
Toy_xmax05 = MC_data.nlargest(int(len(MC_data)*.05), 'xm')

MC_xmax10 = MC_data.nlargest(int(len(MC_data)*.1), 'xm')
Toy_xmax10 = MC_data.nlargest(int(len(MC_data)*.1), 'xm')

MC_xmax15 = MC_data.nlargest(int(len(MC_data)*.15), 'xm')
Toy_xmax15 = MC_data.nlargest(int(len(MC_data)*.15), 'xm')

MC_xmax20 = MC_data.nlargest(int(len(MC_data)*.2), 'xm')
Toy_xmax20 = MC_data.nlargest(int(len(MC_data)*.2), 'xm')

MC_xmax25 = MC_data.nlargest(int(len(MC_data)*.25), 'xm')
Toy_xmax25 = MC_data.nlargest(int(len(MC_data)*.25), 'xm')

MC_xmax30 = MC_data.nlargest(int(len(MC_data)*.3), 'xm')
Toy_xmax30 = MC_data.nlargest(int(len(MC_data)*.3), 'xm')

#%%

for i in range(0,len(columns)):
    para = columns[i]
    plt.hist(MC_xmax01[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='k', label = "1%", density=True)
    plt.hist(MC_xmax05[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='purple', label = "5%", density=True)
    plt.hist(MC_xmax10[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='b', label = "10%", density=True)
    plt.hist(MC_xmax15[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='r', label = "15%", density=True)
    plt.hist(MC_xmax20[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='g', label = "20%", density=True)
    #plt.hist(MC_xmax25[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='orange', label = "25%", density=True)
    #plt.hist(MC_xmax30[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='purple', label = "30%", density=True)
    plt.grid()
    #plt.xlim(0,70)
    #plt.ylim(0,.2)
    plt.legend()
    plt.title("Histogram of "+para+' of most penetrating showers')
    plt.show()
    
#%%
para = 'mir_ngtube'
plt.hist(MC_xmax01[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='k', label = "1%", density=True)
plt.hist(MC_xmax05[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='purple', label = "5%", density=True)
plt.hist(MC_xmax10[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='b', label = "10%", density=True)
plt.hist(MC_xmax15[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='r', label = "15%", density=True)
plt.hist(MC_xmax20[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='g', label = "20%", density=True)
#plt.hist(MC_xmax25[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='orange', label = "25%", density=True)
#plt.hist(MC_xmax30[para], bins = 20, histtype="stepfilled", fill=False, edgecolor='purple', label = "30%", density=True)
plt.grid()
plt.xlim(0, 200)
#plt.ylim(0,.2)
plt.legend()
plt.title("Histogram of "+para+' of most penetrating showers')
plt.show()

#%%
MC_data['dataset'] = 1
toy_data['dataset'] = 0

data = pd.concat([MC_data,toy_data], ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)

Y = data['dataset']
del(data['dataset'])
del(data['mcip'])

# Split dataset into training set and test set
X_tra, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.25)
                                                    # 75% training and 25% test

print(X_tra.iloc[0])

#%% Decision Tree Training
dt = DecisionTreeClassifier(max_leaf_nodes=50,random_state=0)

model = dt.fit(X_tra, y_train)

#Predict the response for test dataset
DTy_pred = model.predict(X_test)

### Evaluating Decision Tree Training
print('Decision Tree:')
#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, DTy_pred))

#Confusion Matrix
cm = confusion_matrix(y_test, DTy_pred)
print(cm)
print([['TP','FN'],['FP','TN']])

#show tree
plt.figure(figsize=(30,30))
plot_tree(dt,proportion=True)
plt.show()

#%% Random Forest Training
dt = RandomForestClassifier(random_state=0)

model = dt.fit(X_tra, y_train)

#Predict the response for test dataset
RFy_pred = model.predict(X_test)

### Evaluating Random Forest Training
print('Random Forest:')
#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, RFy_pred))

#Confusion Matrix
cm = confusion_matrix(y_test, RFy_pred)
print(cm)
print([['TP','FN'],['FP','TN']])


