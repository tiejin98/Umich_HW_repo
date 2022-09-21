import numpy as np
from math import log
from scipy import stats
from sklearn.decomposition import FactorAnalysis


def generate(random_seed = None):
    if random_seed != None:
        np.random.seed(random_seed)
    X = np.random.multivariate_normal(np.zeros(2),np.eye(2),size=100).T
    Gamma = np.zeros((15,2))
    for i in range(15):
        if i <= 7:
            Gamma[i,0] = 1
        else:
            Gamma[i,1] = 1
    W = np.random.multivariate_normal(np.zeros(15),0.5*np.eye(15),size=100).T
    Y = np.dot(Gamma,X)+W
    return Y


def permutation_test(x,num=500):
    n = x.shape[0]
    n_list = list(range(n))
    eigen_ori,_ = np.linalg.eig(np.cov(x,ddof=1))
    eigen_ori.sort(0)
    eigen_ori = eigen_ori[::-1]
    eigen_ori /= np.sum(eigen_ori)
    res = []
    for _ in range(num):
        data = x.copy()
        for i in range(n):
            np.random.shuffle(data[i])
        eigen_shuffle,_ = np.linalg.eig(np.cov(data,ddof=1))
        eigen_shuffle.sort(0)
        eigen_shuffle = eigen_shuffle[::-1]
        eigen_shuffle /= np.sum(eigen_shuffle)
        one_res = 0
        for i in range(n):
            if eigen_shuffle[i] < eigen_ori[i]:
                one_res +=1
        res.append(one_res)
    return np.mean(res)

def simulation_perm(num_per=500,num_sim=100):
    correct = 0
    for _ in range(num_sim):
        y = generate()
        if permutation_test(y,num_per)==2:
            correct += 1
    return correct/num_sim

def choose_from_variance(x):
    eigen_val,_ = np.linalg.eig(np.cov(x,ddof=1))
    eigen_val.sort()
    eigen_val = eigen_val[::-1]
    eigen_val /= np.sum(eigen_val)
    sum = 0
    for i in range(len(eigen_val)):
        sum += eigen_val[i]
        if sum >= 0.9:
            return i+1

def simulation_variance(num_sim=100):
    correct = 0
    for _ in range(num_sim):
        y = generate()
        p = choose_from_variance(y)
        print(p)
        if p == 2:
            correct +=1
    return correct/num_sim

def likelihood_ra_test(x):
    cov_mat = np.cov(x,ddof=1)
    for i in range(x.shape[0]):
        model = FactorAnalysis(i+1).fit(x.T)
        lamb = model.components_.T
        lamb[np.abs(lamb)<0.3] = 0
        psi = np.diagonal(cov_mat - np.dot(lamb,lamb.T))
        psi = np.diag(psi**(-0.5))
        temp = np.dot(psi,cov_mat)
        mat = np.dot(temp,psi)
        eigen_val,_ = np.linalg.eig(mat)
        eigen_val.sort()
        eigen_val = eigen_val[::-1]
        sum = 0
        for j in range(i+1,x.shape[0]):
            sum += eigen_val[j] -1 - log(eigen_val[j])
        sum *= x.shape[1]-1
        freedom = (x.shape[0] - (i+1))**2 -x.shape[0]-(i+1)
        p_value = 1-stats.chi2.cdf(sum,freedom/2)
        if p_value >0.05 or freedom <= 0:
            return i+1

def simulation_lrt(num_sim=100):
    corr = 0
    for _ in range(num_sim):
        y = generate()
        if likelihood_ra_test(y) == 2:
            corr += 1
    return corr/num_sim

print(simulation_perm())
print(simulation_variance())
print(simulation_lrt())
