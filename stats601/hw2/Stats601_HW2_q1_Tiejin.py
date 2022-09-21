import numpy as np
import pandas as pd
from copy import deepcopy

data = pd.read_csv("cytometry-1.txt", delimiter=" ").values
cov_mat = np.cov(data.T)
adj_mat = np.zeros((11, 11))
one_ele = [(0, 1), (1, 3), (1, 6), (1, 7), (1, 9), (2, 3), (2, 7), (2, 9), (3, 6), (3, 7), (3, 9), (3, 10),
           (6, 9), (7, 9), (8, 9), (9, 10)]
for pair in one_ele:
    adj_mat[pair[0], pair[1]] = 1
    adj_mat[pair[1], pair[0]] = 1
for i in range(data.shape[1]):
    adj_mat[i, i] = 1
p = list(range(11))
w = cov_mat


def recover_beta(beta_star, no_edge, p_use, n):
    p = len(no_edge) + len(p_use)
    beta = np.zeros(p)
    for i in range(len(p_use)):
        if p_use[i] < n:
            beta[p_use[i]] = beta_star[i]
        else:
            beta[p_use[i] - 1] = beta_star[i]
    return beta


def one_round_iter(data, w, S, p):
    w12_res = []
    beta_res = []
    w_res = np.zeros_like(w)
    for i in range(data.shape[1]):
        w_res[i, i] = S[i, i]
        p_use = deepcopy(p)
        p_use.remove(i)
        w11 = w[p_use]
        w11 = w11[:, p_use]
        no_edge = []
        for j in range(data.shape[1]):
            if adj_mat[i, j] == 0:
                no_edge.append(j)
        for num in no_edge:
            p_use.remove(num)
        if len(p_use) == 0:
            beta = np.zeros(data.shape[1] - 1)
        else:
            w11_star = w[p_use]
            w11_star = w11_star[:, p_use]
            s12 = S[i]
            s12_star = s12[p_use]
            if len(p_use) == 1:
                beta_star = np.squeeze((1 / w11_star) * s12_star, 1)
            elif len(p_use) > 1:
                beta_star = np.dot(np.linalg.inv(w11_star), s12_star)
            beta = recover_beta(beta_star, no_edge, p_use, i)
        w12 = np.dot(w11, beta)
        for j in range(len(w12)):
            if j < i:
                w_res[i, j] = w12[j]
            else:
                w_res[i, j + 1] = w12[j]
        w12_res.append(w12)
        beta_res.append(beta)
    return w_res, w12_res, beta_res


def estimation(data, s, p, eps):
    w_former = s
    w_now, w12_res, beta_res = one_round_iter(data, s, s, p)
    change = np.sum(np.abs(w_now - w_former)) / (w_former.shape[0] * w_former.shape[1])
    n = 1
    while change > eps:
        w_former = w_now
        w_now, w12_res, beta_res = one_round_iter(data, w_former, s, p)
        change = np.sum(np.abs(w_now - w_former)) / (w_former.shape[0] * w_former.shape[1])
        n += 1
    theta = np.zeros_like(s)
    for i in range(data.shape[1]):
        theta[i, i] = 1 / (s[2, 2] - np.dot(w12_res[i].T, beta_res[i]))
        theta12 = -theta[i, i] * beta_res[i]
        for j in range(len(theta12)):
            if j < i:
                theta[i, j] = theta12[j]
            else:
                theta[i, j + 1] = theta12[j]
    return theta,w_now


theta,w_now = estimation(data, w, p, 0.1)
print(theta)
print(w_now)

