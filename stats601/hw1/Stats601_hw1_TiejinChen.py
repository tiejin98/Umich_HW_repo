import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy import stats
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from sklearn import preprocessing


data = pd.read_csv("heightWeightData.txt",header=None,names=["gender","weight","height"])
female_data = data[data['gender']==2][['weight','height']]
mean_vector = np.mean(female_data.values,axis=0)
sigma = np.cov(female_data.values.T,ddof=1)
def draw_elip_sca(mean,cov,x,y,confi=5.991):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lambda_, v= np.linalg.eig(cov)
    sqrt_lambda = np.sqrt(np.abs(lambda_))
    width = 2*np.sqrt(confi)*sqrt_lambda[0]
    height = 2*np.sqrt(confi)*sqrt_lambda[1]
    angle = np.rad2deg(np.arccos(v[0,0]))
    elip = Ellipse(xy=mean,width=width,height=height,angle=angle,alpha=0.5)
    ax.add_patch(elip)
    plt.scatter(x,y,color='red')
    plt.show()
draw_elip_sca(mean_vector,sigma,female_data['weight'],female_data['height'])
# print(mean_vector,sigma)
#
stand_data = preprocessing.scale(female_data.values,axis=0)
mean_vector_stand = np.mean(stand_data,axis=0)
sigma_stand = np.cov(stand_data.T,ddof=0)
draw_elip_sca(mean_vector_stand,sigma_stand,stand_data[:,0],stand_data[:,1])
# print(sigma_stand)

lambda_,v = np.linalg.eig(sigma_stand)
lambda_ = 1/lambda_
lamb_mat = np.diag(np.sqrt(lambda_))
whiten_data_pre = np.matmul(lamb_mat,v.T)
whiten_data = np.matmul(whiten_data_pre,stand_data.T).T
mean_vector_whiten = np.mean(whiten_data,axis=0)
sigma_whiten = np.cov(whiten_data.T,ddof=0)
print(sigma_whiten)
print(mean_vector_whiten)
draw_elip_sca(mean_vector_whiten,sigma_whiten,whiten_data[:,0],whiten_data[:,1])

male_data = data[data['gender']==1][['weight','height']]
mean_vector_male = np.mean(male_data.values,axis=0)
n1 = 137
n2 = 73
minus = mean_vector-mean_vector_male
sigma_male = np.cov(male_data.values.T,ddof=1)
sigma_pooled = ((n1-1)*sigma+(n2-1)*sigma_male)/(n1+n2-2)
sigma_pooled_inverse = np.linalg.inv(sigma_pooled)
temp_t2 = np.matmul(minus,sigma_pooled_inverse)
t2 = n1*n2*np.matmul(temp_t2,minus.T)/(n1+n2)
f_stat = t2*(n1+n2-2-1)/((n1+n2-2)*2)
f_dis = stats.f(2,n1+n2-2-1)
print(1-f_dis.cdf(f_stat))

def one_simulation(p):
    mean_vector = np.zeros(p)
    cov_matrix_temp = np.random.uniform(0,10,size=(p,p))
    cov_matrix = np.matmul(cov_matrix_temp.T, cov_matrix_temp)
    sample_result = np.random.multivariate_normal(mean_vector,cov_matrix,100)
    xn_bar = np.mean(sample_result,axis=0)
    sn = np.cov(sample_result.T,ddof=1)
    sn_inverse = np.linalg.inv(sn)
    T2 = 100 * np.matmul(np.matmul(xn_bar, sn_inverse), xn_bar.T)
    p_value = 1-stats.chi2(p).cdf(T2)
    if p_value <=0.05:
        return 1
    else:
        return 0


def simulation(n,p,random_seed=42):
    np.random.seed(random_seed)
    res = []
    for i in range(n):
        res.append(one_simulation(p))
    return sum(res)/n


def high_one_simulation(p):
    mean_vector = np.zeros(p)
    cov_matrix_temp = np.random.uniform(0,10,size=(p,p))
    cov_matrix = np.matmul(cov_matrix_temp.T, cov_matrix_temp)
    sample_result = np.random.multivariate_normal(mean_vector,cov_matrix,100)
    xn_bar = np.mean(sample_result,axis=0)
    sn = np.cov(sample_result.T,ddof=1)
    var_vector = np.diagonal(sn)
    ds_half = np.diag(var_vector ** (-0.5))
    ds_inverse = np.diag(var_vector**(-1))
    R = np.matmul(np.matmul(ds_half, sn), ds_half)
    cpn = 1 + (np.trace(np.matmul(R,R))) / (p ** (1.5))
    nomi = 100*np.matmul(np.matmul(xn_bar,ds_inverse),xn_bar.T) - (99*p)/97
    demoni = 2*(np.trace(np.matmul(R,R))-(p*p/99)) *cpn
    T1 = nomi/(demoni**0.5)
    p_value = min(stats.norm.cdf(T1),1-stats.norm.cdf(T1))
    if p_value <=0.05:
        return 1
    else:
        return 0

def high_simulation(n,p,random_seed=42):
    np.random.seed(random_seed)
    res = []
    for i in range(n):
        res.append(high_one_simulation(p))
    return sum(res)/len(res)


print("p:3,Type I error:",simulation(2000,3))
print("p:10,Type I error:",simulation(2000,10))
print("p:40,Type I error:",simulation(2000,40))
print("p:80,Type I error:",simulation(2000,80))
print(high_simulation(500,150))


