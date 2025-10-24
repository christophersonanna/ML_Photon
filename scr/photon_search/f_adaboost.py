# -*- coding: utf-8 -*-

from sklearn import metrics
from sklearn.tree import AdaBoostClassifier
from sklearn.metrics import confusion_matrix

from b_data_organization import Photons, Not_photons
from c_split_data import y_test, y_train, X_train, X_test

### Adaboosting Training
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=5,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
ABCy_pred = model.predict(X_test)

### Evalutating Adaboost training
print('Adaboost Results:')
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, ABCy_pred))

# Model Accuracy, how often is the classifier correct?
print("Loss:",metrics.zero_one_loss(y_test, ABCy_pred))

#Confusion Matrix
cm = confusion_matrix(y_test, ABCy_pred)
print(cm)
print([['TP','FN'],['FP','TN']])

###adaboost by hand evaluation
ABC_correct_photon = 0
ABC_correct_notphoton = 0
ABC_incorrect_photon = 0
ABC_incorrect_notphoton = 0

for i in range(0, len(y_test)):
    if y_test[i]==ABCy_pred[i]:
        if ABCy_pred[i]==1:
            ABC_correct_photon += 1
        else:
            ABC_correct_notphoton += 1
    else:
        if ABCy_pred[i]==0:
            ABC_incorrect_photon += 1
        else:
            ABC_incorrect_notphoton += 1
            
print('Adaboost')
print('c_p: ', ABC_correct_photon, ',',ABC_correct_photon/24960*100)
print('c_np: ', ABC_correct_notphoton, ',',ABC_correct_notphoton/25017*100)
print('ic_p: ', ABC_incorrect_photon, ',',ABC_incorrect_photon/24960*100)
print('ic_np: ', ABC_incorrect_notphoton, ',',ABC_incorrect_notphoton/25017*100)

print('\nTrue Photons: ',Photons, 'Added: ',ABC_correct_photon+ABC_incorrect_photon)
print('True Not Photons: ',Not_photons, 'Added: ',ABC_correct_notphoton+ABC_incorrect_notphoton)
