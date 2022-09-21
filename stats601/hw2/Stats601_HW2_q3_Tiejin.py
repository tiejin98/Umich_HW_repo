import pyreadr
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

data =pyreadr.read_r("nytimes.RData")
df = data['nyt.frame']
features = df.iloc[:,1:].values
map_dict = {'art':0,'music':1}
df['class.labels'] = df["class.labels"].map(map_dict)
df_data = df.iloc[:,1:]
labels = df["class.labels"].values
def data_normlize(x):
    for i in range(x.shape[1]):
        x[:,i] = x[:,i]-np.mean(x[:,i])
    return x

#features = data_normlize(features)

def gram_schmidt(x):
    v = np.zeros_like(x)
    for i in range(x.shape[1]):
        if i == 0:
            v[:,i] = x[:,i].copy()
        else:
            vi = x[:,i].copy()
            for j in range(i):
                vi -= (np.dot(x[:,i],v[:,j])/np.dot(v[:,j],v[:,j]))*v[:,j]
            v[:,i] = vi
    return v


def qr_decom(x):
    orth_x = gram_schmidt(x)
    q = np.zeros_like(orth_x)
    for i in range(orth_x.shape[1]):
        factor = sum(orth_x[:,i]**2)**(0.5)
        q[:,i] = orth_x[:,i]/factor
    r = np.dot(q.T,x)
    # for i in range(r.shape[0]):
    #     for j in range(r.shape[1]):
    #         if j<i and r[i,j] <=1.12e-16:
    #             r[i,j] = 0
    return q,r

def eigen_decom(x,iter_num=10):
    A = x.copy()
    for _ in tqdm(range(iter_num)):
        qi,ri = np.linalg.qr(A)
        A = np.dot(ri,qi)
    return A,np.linalg.inv(x-A)

def eigen_decom_test(x):
    A=x.copy()
    n = 0
    while True:
        qi,ri = np.linalg.qr(A)
        A = np.dot(ri,qi)
        n += 1
        if n %10 == 0:
            print(n)
        try:
            vect = np.linalg.inv(x-A)
            break
        except:
            continue
    return A,vect

def PCA_direction(x ,Loading = False):
    cov_mat = np.cov(x.T,ddof=1)
    eigen_vals,eigen_vectors = np.linalg.eig(cov_mat)
    eigen_val_index = np.argsort(eigen_vals)[::-1]
    eigen_vals = eigen_vals[eigen_val_index]
    eigen_vectors = eigen_vectors[:,eigen_val_index]
    if Loading:
        return eigen_vals.real,eigen_vectors.real
    else:
        return eigen_vectors

def PCA(x,dim):
    U = PCA_direction(x)[:,:dim]
    return np.dot(x,U).real

def draw(x,label,dim):
    data = PCA(x,dim)
    data1 = data[label==0]
    data2 = data[label==1]
    if dim == 1:
        ax = plt.subplot()
        ax.scatter(data1,np.zeros_like(data1),color="r",label="art")
        ax.scatter(data2,np.zeros_like(data2),color="b",label="music")
    elif dim == 2:
        ax = plt.subplot()
        ax.scatter(data1[:,0],data1[:,1],color="r",label="art")
        ax.scatter(data2[:,0],data2[:,1],color="b",label="music")
    elif dim == 3:
        ax = plt.subplot(projection="3d")
        ax.scatter(data1[:,0],data1[:,1],data1[:,2],color="r",label="art")
        ax.scatter(data2[:,0],data2[:,1],data2[:,2],color="b",label="music")
    plt.legend()
    plt.show()



draw(features,labels,1)
draw(features,labels,2)
draw(features,labels,3)

val,vectors = PCA_direction(features,Loading=True)
print(val[0]/np.sum(val))
print(val[1]/np.sum(val))
print(val[2]/np.sum(val))

u1 = vectors[:,0]
u2 = vectors[:,1]
u3 = vectors[:,2]
U = [u1,u2,u3]
for u in U:
    index = np.argsort(u)
    neg_index = index[:20]
    pos_index = index[-20:]
    pos_index = list(pos_index)
    pos_index.reverse()
    neg_words = df_data.columns[neg_index]
    pos_words = df_data.columns[pos_index]
    print("pos_words:",pos_words)
    print("neg_words:",neg_words)