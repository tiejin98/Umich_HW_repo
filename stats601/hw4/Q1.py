import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from statsmodels.multivariate import factor

from statsmodels.multivariate.factor import Factor,FactorResults

def generate(random_seed = None):
    if random_seed != None:
        np.random.seed(random_seed)
    X = np.random.multivariate_normal(np.zeros(3),np.eye(3),size=200).T
    Gamma = np.diag([1,0.001,10])
    Gamma[1,0] = 1
    print(Gamma.dot(Gamma.T))
    Y = np.dot(Gamma,X)
    return (Y,X)

Y,X = generate(100)
Y= Y.T
model =PCA(n_components=1)
model.fit(Y)
print(model.components_)
fa = Factor(Y,n_factor=1)
print(fa.fit().loadings)
fa_res = FactorResults(fa)
print(fa_res.get_loadings_frame().data)