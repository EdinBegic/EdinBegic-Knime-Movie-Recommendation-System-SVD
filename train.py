import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import getcwd

def avg(R):
    mu = 0
    cnt = 0
    for x in R:
        for y in x:
            if y > 0:
                cnt = cnt + 1
                mu += y
    return mu / cnt


def item_bias(R, D, U, mu):
    bq = np.zeros([D])  # Movie bias
    lambda1 = 25

    for i in range(D):
        Ri = 0
        s = 0
        for u in range(U):
            if R[u][i] > 0:
                Ri = Ri + 1
                s = s + R[u][i] - mu
        bq[i] = s / (lambda1 + Ri)

    return bq


def user_bias(R, D, U, mu, bq):
    bp = np.zeros([U])  # User bias
    lambda2 = 10
    for u in range(U):
        Ru = 0
        s = 0
        for i in range(D):
            if R[u][i] > 0:
                Ru = Ru + 1
                s = s + R[u][i] - mu - bq[i]
        bp[u] = s / (lambda2 + Ru)

    return bp


def get_bias(R, D, U):
    mu = avg(R)

    bq = item_bias(R, D, U, mu)
    bp = user_bias(R, D, U, mu, bq)

    B = np.zeros([U, D])
    for i in range(0, U):
        for j in range(0, D):
            B[i][j] = mu + bq[j] + bp[i]
    return B


def get_biased_error(R, W, X, Y, B):
    return np.sum((W * (R - B - np.dot(X, Y)) ** 2))


def biased_als(R, W, K=30, steps=10, R_test=None, W_test=None, Y=None):
    if (R_test is None) ^ (W_test is None):
        raise ValueError('R_test and W_test have to be either None or not None')
    elif R_test is not None:
        W_test = W_test.astype(np.float64, copy=False)
        R_test = R_test.astype(np.float64, copy=False)

    fix_movies = False
    U, D = R.shape

    if Y is not None:
        fix_movies = True
        K, _ = Y.shape
    else:
        Y = 5 * np.random.rand(K, D).astype(np.float64, copy=False)

    W = W.astype(np.float64, copy=False)
    # W = np.vstack((np.ones(W.shape[1]), W))
    # W = np.hstack((np.ones((W.shape[0], 1)), W))
    R = R.astype(np.float64, copy=False)
    X = 5 * np.random.rand(U, K).astype(np.float64, copy=False)
    B = get_bias(R, D, U).astype(np.float64, copy=False)
    error_log = []
    error_test_log = []
    _lambda = 0.05

    R = R - B

    beta = np.random.rand(U, 1)
    gamma = np.random.rand(1, D)

    err = np.inf
    while steps > 0 and err > 0.002:
        _X = np.hstack((np.ones((U, 1)), X))
        _Y = np.vstack((gamma, Y))

        for i in range(D):
            _Y[:, i] = findY(K, R, W, _X, _lambda, i)

        gamma = _Y[0]

        _Y[0] = np.ones((1, Y.shape[1]))
        _X[:, [0]] = beta

        for u in range(U):
            _X[u] = findX(K, R, W, _X, _Y, _lambda, u)

        beta = _X[:, [0]]
        X = _X[:, 1:]
        Y = _Y[1:, :]

        for i in range(0, U):
            for j in range(0, D):
                B[i][j] = gamma[j] + beta[i][0]
        err = get_biased_error(R, W, X, Y, B)
        error_log.append(err)
        print('Error: {}'.format(err))
        if R_test is not None:
            err_test = get_biased_error(R_test, W_test, X, Y, B)
            error_test_log.append(err_test)
            print('Test Error: {}'.format(err_test))

        steps = steps - 1

    plt.plot(error_log)
    if R_test is not None:
        plt.plot(error_test_log, 'r')
    plt.title('Learning RMSE')
    plt.xlabel('Iteration count')
    plt.ylabel('Error')
    plt.show(block=False)
    plt.pause(5)
    plt.close('all')

    return X, Y, B


def findX(K, R, W, _X, _Y, _lambda, u):
    Wu = np.diag(W[u])
    return np.linalg.solve(np.dot(_Y, np.dot(Wu, _Y.T)) + _lambda * np.eye(K + 1),
                           np.dot(_Y, np.dot(Wu, R[u].T))).T


def findY(K, R, W, _X, _lambda, i):
    Wi = np.diag(W.T[i])
    return np.linalg.solve(np.dot(_X.T, np.dot(Wi, _X)) + _lambda * np.eye(K + 1),
                           np.dot(_X.T, np.dot(Wi, R[:, i])))


def main():
    R = pd.read_csv(getcwd() + "/data/R.csv")
    W = pd.read_csv(getcwd() + "/data/W.csv")

    R.drop(columns=R.columns[0], axis=1, inplace=True)
    W.drop(columns=W.columns[0], axis=1, inplace=True)

    X, Y, B = biased_als(R.values, W.values)

    pd.DataFrame(np.dot(X, Y) + B).to_csv(getcwd() + "/predictions.csv")

main()

