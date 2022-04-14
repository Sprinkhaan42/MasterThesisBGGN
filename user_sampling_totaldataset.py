# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:00:14 2022

@author: orteg
"""

import numpy as np
prng = np.random.RandomState(123)
from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing

NUM_BUNDLEITEM_SAMPLING = 1500

#### TRAIN DATA ####

filename_train = 'user_bundle_original.txt'
with open(filename_train) as f:
    lines =  list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
  

users = np.asarray([lines[i][0] for i in range(len(lines))])
bundles = np.asarray([lines[i][1] for i in range(len(lines))])

## Sampling ##
ix_10perc_users = np.where(np.asarray(users) <= 400)[0]
ix_10perc_bundles = np.where(np.asarray(bundles) <= 400)[0]

ix_10perc_userbundles= np.intersect1d(ix_10perc_users,ix_10perc_bundles)

## Re-scaling

# Users
le_users = preprocessing.LabelEncoder()
users_idrescaled = le_users.fit_transform(users[ix_10perc_userbundles])

# Bundles
le_bundles = preprocessing.LabelEncoder()
original_bundles_selected = bundles[ix_10perc_userbundles]
bundles_idrescaled = le_bundles.fit_transform(bundles[ix_10perc_userbundles])

user_bundles_idrescaled = np.column_stack((users_idrescaled, bundles_idrescaled))

user_bundle_tuple_list = [tuple(i) for i in user_bundles_idrescaled.tolist()]
user_bundle_sorted_tuple_list = sorted(user_bundle_tuple_list, key=lambda x: x[0])

rs = ShuffleSplit(n_splits = 1, test_size = 0.20, random_state = 123)
ix_tr, ix_ts = [(a, b) for a, b in rs.split(user_bundle_sorted_tuple_list)][0]

user_bundle_train = np.asarray(user_bundle_sorted_tuple_list)[ix_tr]
user_bundle_test = np.asarray(user_bundle_sorted_tuple_list)[ix_ts]

user_bundle_train_tuple_list = [tuple(i) for i in user_bundle_train.tolist()]
user_bundle_test_tuple_list = [tuple(i) for i in user_bundle_test.tolist()]

user_bundle_train_sorted_tuple_list = sorted(user_bundle_train_tuple_list, key=lambda x: x[0])
user_bundle_test_sorted_tuple_list = sorted(user_bundle_test_tuple_list, key=lambda x: x[0])

### User_Bundle ###

f = open('user_bundle.txt', 'w')
for t in user_bundle_sorted_tuple_list:
    line = '\t'.join(str(x) for x in t)
    f.write(line + '\n')
f.close()

### Train User_Bundle ###

f = open('user_bundle_train.txt', 'w')
for t in user_bundle_train_sorted_tuple_list:
    line = '\t'.join(str(x) for x in t)
    f.write(line + '\n')
f.close()

### Test User_Bundle ###

f = open('user_bundle_test.txt', 'w')
for t in user_bundle_test_sorted_tuple_list:
    line = '\t'.join(str(x) for x in t)
    f.write(line + '\n')
f.close()

### Bundle Item ####
filename = 'bundle_item_original.txt'
with open(filename) as f:
    bundle_item =  list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
    
items = np.asarray([bundle_item[i][1] for i in range(len(bundle_item))])
bundles = np.asarray([bundle_item[i][0] for i in range(len(bundle_item))])

ix_bundles = np.where(np.isin(np.asarray(bundles), original_bundles_selected))[0]

bundles = bundles[ix_bundles]
items = items[ix_bundles]

rescaled_bundles = le_bundles.transform(bundles)

le_items = preprocessing.LabelEncoder()
rescaled_items = le_items.fit_transform(items)

bundle_item_list = [(rescaled_bundles[i], rescaled_items[i]) for i in range(len(rescaled_bundles))]

f = open('bundle_item.txt', 'w')
for t in bundle_item_list:
    line = '\t'.join(str(x) for x in t)
    f.write(line + '\n')
f.close()
