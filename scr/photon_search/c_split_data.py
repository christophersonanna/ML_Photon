# -*- coding: utf-8 -*-

from a_get_data import X_test, X_train

# Split dataset into training set and test set
y_train = X_train['photon']
y_test = X_test['photon']

del(X_test['photon'])
del(X_train['photon'])


