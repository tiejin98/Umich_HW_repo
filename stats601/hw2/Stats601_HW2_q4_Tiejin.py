import numpy as np
from sklearn.decomposition import FactorAnalysis,PCA
import matplotlib.pyplot as plt

def generate(random_seed = None):
    if random_seed != None:
        np.random.seed(random_seed)
    X = np.random.multivariate_normal(np.zeros(2),np.eye(2),size=100).T
    Gamma = np.ones((7,2))
    for i in range(15):
        if i <= 2:
            Gamma[i,1] = 0
        elif i >= 4:
            Gamma[i,0] = 1
    W = np.random.multivariate_normal(np.zeros(7),0.4*np.eye(7),size=100).T
    Y = np.dot(Gamma,X)+W
    return Y

y =generate()
X = FactorAnalysis(2).fit_transform(y.T)
ax = plt.subplot()
ax.scatter(X[:,0],X[:,1])
plt.show()

pca_x =PCA(2).fit_transform(y.T)
ax = plt.subplot()
ax.scatter(X[:,0],X[:,1],color = "b",label="Factor")
ax.scatter(pca_x[:,0],pca_x[:,1],color = "r",label="PCA")
plt.legend()
plt.show()
