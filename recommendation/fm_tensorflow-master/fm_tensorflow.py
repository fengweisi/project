from __future__ import print_function
from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm


def vectorize_dic(dic, ix=None, p=None):
    if (ix is None):
        d = count(0)
        ix = defaultdict(lambda: next(d))

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

    if (p is None):
        p = len(ix)

    ixx = np.where(col_ix < p)

    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])),
                          shape=(n, p)), ix


def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError(
            'Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)


def network(X_train):
    n, p = X_train.shape

    k = 10

    X = tf.placeholder('float', shape=[None, p])
    y = tf.placeholder('float', shape=[None, 1])

    w0 = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.zeros([p]))
    V = tf.Variable(tf.random_normal([k, p], stddev=0.01))

    y_hat = tf.Variable(tf.zeros([n, 1]))

    linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X),
                                            1,
                                            keepdims=True))
    pair_interactions = tf.multiply(
        0.5,
        tf.reduce_sum(tf.subtract(
            tf.pow(tf.matmul(X, tf.transpose(V)), 2),
            tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
                      1,
                      keepdims=True))
    y_hat = tf.add(linear_terms, pair_interactions)
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')
    l2_norm = tf.add(tf.reduce_sum(tf.multiply(lambda_w, tf.pow(W, 2))),
                     tf.reduce_sum(tf.multiply(lambda_v, tf.pow(V, 2))))
    error = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
    loss = tf.add(error, l2_norm)
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=0.01).minimize(loss)
    return X, y, error, optimizer


def train_net(X_train, y_train, X_test, y_test):
    X, y, error, optimizer = network(X_train)
    epochs = 10
    batch_size = 1000

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    for epoch in tqdm(range(epochs), unit='epoch'):
        perm = np.random.permutation(X_train.shape[0])
        for bX, bY in batcher(X_train[perm], y_train[perm], batch_size):
            sess.run(optimizer,
                     feed_dict={
                         X: bX.reshape(-1, X_train.shape[1]),
                         y: bY.reshape(-1, 1)
                     })
    errors = []
    for bX, bY in batcher(X_test, y_test):
        errors.append(
            sess.run(error,
                     feed_dict={
                         X: bX.reshape(-1, X_train.shape[1]),
                         y: bY.reshape(-1, 1)
                     }))

    RMSE = np.sqrt(np.array(errors).mean())
    print(RMSE)

    sess.close()


cols = ['user', 'item', 'rating', 'timestamp']
train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)
test = pd.read_csv('data/ua.test', delimiter='\t', names=cols)

X_train, ix = vectorize_dic({
    'users': train.user.values,
    'items': train.item.values
})
X_test, ix = vectorize_dic(
    {
        'users': test.user.values,
        'items': test.item.values
    }, ix, X_train.shape[1])
y_train = train.rating.values
y_test = test.rating.values
X_train = X_train.todense()
X_test = X_test.todense()

print(X_train.shape)
print(X_test.shape)

train_net(X_train, y_train, X_test, y_test)
