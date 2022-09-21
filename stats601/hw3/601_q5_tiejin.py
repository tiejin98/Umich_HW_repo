import pyreadr
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA



data =pyreadr.read_r("nytimes.RData")
df = data['nyt.frame']
features = df.iloc[:,1:].values
map_dict = {'art':0,'music':1}
df['class.labels'] = df["class.labels"].map(map_dict)
df_data = df.iloc[:,1:]
label = df["class.labels"].values

dim = [1,2,3]
for d in dim:
    model_poly = KernelPCA(d,kernel="poly",degree=3)
    model_gauss = KernelPCA(d,kernel="rbf")
    res_poly = model_poly.fit_transform(features)
    res_gauss = model_gauss.fit_transform(features)
    if d == 1:
        data1 = res_poly[label==0]
        data2 = res_poly[label==1]
        ax = plt.subplot()
        ax.scatter(data1,np.zeros_like(data1),color="r",label="art")
        ax.scatter(data2,np.zeros_like(data2),color="b",label="music")
        plt.title("polynomial kernel")
        plt.legend()
        plt.show()
        data1 = res_gauss[label==0]
        data2 = res_gauss[label==1]
        ax1 = plt.subplot()
        ax1.scatter(data1,np.zeros_like(data1),color="r",label="art")
        ax1.scatter(data2,np.zeros_like(data2),color="b",label="music")
        plt.title("Gaussian kernel")
        plt.legend()
        plt.show()
    elif d == 2:
        data1 = res_poly[label==0]
        data2 = res_poly[label==1]
        ax = plt.subplot()
        ax.scatter(data1[:,0],data1[:,1],color="r",label="art")
        ax.scatter(data2[:,0],data2[:,1],color="b",label="music")
        plt.title("polynomial kernel")
        plt.legend()
        plt.show()
        data1 = res_gauss[label==0]
        data2 = res_gauss[label==1]
        ax1 = plt.subplot()
        ax1.scatter(data1[:,0],data1[:,1],color="r",label="art")
        ax1.scatter(data2[:,0],data2[:,1],color="b",label="music")
        plt.title("Gaussian kernel")
        plt.legend()
        plt.show()
    elif d == 3:
        data1 = res_poly[label==0]
        data2 = res_poly[label==1]
        ax = plt.subplot(projection="3d")
        ax.scatter(data1[:,0],data1[:,1],data1[:,2],color="r",label="art")
        ax.scatter(data2[:,0],data2[:,1],data2[:,2],color="b",label="music")
        plt.title("polynomial kernel")
        plt.legend()
        plt.show()
        data1 = res_gauss[label==0]
        data2 = res_gauss[label==1]
        ax1 = plt.subplot(projection="3d")
        ax1.scatter(data1[:,0],data1[:,1],data1[:,2],color="r",label="art")
        ax1.scatter(data2[:,0],data2[:,1],data2[:,2],color="b",label="music")
        plt.title("Gaussian kernel")
        plt.legend()
        plt.show()


