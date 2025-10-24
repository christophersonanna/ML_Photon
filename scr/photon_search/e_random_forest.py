# -*- coding: utf-8 -*-

from sklearn import metrics
from sklearn.tree import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from b_data_organization import Photons, Not_photons
from c_split_data import y_test, y_train, X_train, X_test

# Random Forest Training
dt = RandomForestClassifier(random_state=0)

model = dt.fit(X_train, y_train)

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

###random forest by hand evaluation
RF_correct_photon = 0
RF_correct_notphoton = 0
RF_incorrect_photon = 0
RF_incorrect_notphoton = 0

for i in range(0, len(y_test)):
    if y_test[i]==RFy_pred[i]:
        if RFy_pred[i]==1:
            RF_correct_photon += 1
        else:
            RF_correct_notphoton += 1
    else:
        if RFy_pred[i]==0:
            RF_incorrect_photon += 1
        else:
            RF_incorrect_notphoton += 1
            
print('Random Forest')
print('c_p: ', RF_correct_photon, ',',RF_correct_photon/24960*100)
print('c_np: ', RF_correct_notphoton, ',',RF_correct_notphoton/25017*100)
print('ic_p: ', RF_incorrect_photon, ',',RF_incorrect_photon/24960*100)
print('ic_np: ', RF_incorrect_notphoton, ',',RF_incorrect_notphoton/25017*100)

print('\nTrue Photons: ',Photons, 'Added: ',RF_correct_photon+RF_incorrect_photon)
print('True Not Photons: ',Not_photons, 'Added: ',RF_correct_notphoton+RF_incorrect_notphoton)
