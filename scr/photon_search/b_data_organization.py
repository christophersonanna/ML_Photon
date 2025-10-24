# -*- coding: utf-8 -*-

from a_get_data import X_test, X_train


# dataset breakdown
Photons = 0
Not_photons = 0

for i in range(0,len(X_test)):
    if X_test['photon'][i] == 1:
        Photons += 1
    else:
        Not_photons += 1
        
print('Photons: ', Photons)
print('Not Photons: ', Not_photons)


"""
one = 492175
two = 254146
three = 99975
four = 432526
five = 356519

print('1: ',one/(one+two+three+four+five)*100)
print('2: ',two/(one+two+three+four+five)*100)
print('3: ',three/(one+two+three+four+five)*100)
print('4: ',four/(one+two+three+four+five)*100)
print('5: ',five/(one+two+three+four+five)*100)
"""