# -*- coding: utf-8 -*-

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from b_data_organization import Photons, Not_photons
from c_split_data import y_test, y_train, X_train, X_test

print(X_train.columns)
#Decision Tree
dt = DecisionTreeClassifier(random_state=0)

model = dt.fit(X_train, y_train)

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

###decision tree by hand evaluation
DT_correct_photon = 0
DT_correct_notphoton = 0
DT_incorrect_photon = 0
DT_incorrect_notphoton = 0

for i in range(0, len(y_test)):
    if y_test[i]==DTy_pred[i]:
        if DTy_pred[i]==1:
            DT_correct_photon += 1
        else:
            DT_correct_notphoton += 1
    else:
        if DTy_pred[i]==0:
            DT_incorrect_photon += 1
        else:
            DT_incorrect_notphoton += 1
            
print('Decision Tree')
print('c_p: ', DT_correct_photon, ',',DT_correct_photon/24960*100)
print('c_np: ', DT_correct_notphoton, ',',DT_correct_notphoton/25017*100)
print('ic_p: ', DT_incorrect_photon, ',',DT_incorrect_photon/24960*100)
print('ic_np: ', DT_incorrect_notphoton, ',',DT_incorrect_notphoton/24960*100)

print('\nTrue Photons: ',Photons, 'Added: ',DT_correct_photon+DT_incorrect_photon)
print('True Not Photons: ',Not_photons, 'Added: ',DT_correct_notphoton+DT_incorrect_notphoton)
