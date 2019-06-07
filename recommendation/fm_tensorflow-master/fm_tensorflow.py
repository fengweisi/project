from itertools import count 
from collections import defaultdict
from scipy.sparse import csr
#from __future__ import print_function

def vectorize_dic(dic, ix=None, p=None):
	if (ix == None):
		d = count(0)
		ix = defaultdict(lambda:next(d))

	n = len(list(dic.values())[0])
	g = len(list(dic.keys()))
	nz = n * g
	col_ix = np.empty(nz, dtype=int)
	i = 0
	for k, lis in dic.items():
		col_ix[i::g] = [ix[str(el) + str(k)] for el in lis]
		i += 1

	row_ix = np.repeat(np.arange(0, n), g)
	data = np.ones(nz)

	if (p == None):
		p = len(ix)

	ixx = np.where(col_ix < p)


import pandas as pd 
import numpy as np 
from sklearn.feature_extraction import DictVectorizer 

cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('data/ua.test', delimiter='\t', names=cols)

X_train, ix = vectorize_dic({'users' : train.user.values, 'items' : train.item.values})
X_test, ix = vectorize_dic({'users' : test.user.values, 'items' : test.item.values})
y_train = train.rating.values
y_test = test.rating.values 

X_train = X_train.todense()
X_test = X_test.todense()

import tensorflow as tf 

n, p = X_train.shape

k = 10
X = tf.placeholder('float')