import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate(random_seed = None):
    if random_seed != None:
        np.random.seed(random_seed)
    X = np.random.multivariate_normal(np.zeros(2),np.eye(2),size=100).T
    Gamma = np.ones((7,2))
    for i in range(7):
        if i <= 2:
            Gamma[i,1] = 0
        elif i >= 4:
            Gamma[i,0] = 0
    W = np.random.multivariate_normal(np.zeros(7),0.4*np.eye(7),size=100).T
    Y = np.dot(Gamma,X)+W
    return (Y,X)




#
# def em(y,iter,L,phi):
#     n,q = y.shape
#     mu = np.mean(y, axis=0)
#     for _ in range(iter):
#         Exi_res = 0
#         Exixit_res = 0
#         S_res = 0
#         for i in range(n):
#             A = np.linalg.inv(np.eye(2)+ np.dot(np.dot(L.T,np.linalg.inv(phi)),L))
#             temp = np.dot(np.dot(A,L.T),np.linalg.inv(phi))
#             x = np.dot(temp,y[i]-mu).reshape(-1,1)
#             Exi_res += np.dot((y[i]-mu).reshape(-1,1),x.T)
#             Exixit_res += np.dot(x,x.T) + A
#             S_res += np.dot(y[i].reshape(-1,1),y[i].reshape(-1,1).T) - np.dot(np.dot(y[i].reshape(-1,1),x.T),L.T)
#             S_res -= np.dot(np.dot(L,x),y[i].reshape(-1,1).T) - np.dot(np.dot(L,np.dot(x,x.T)+A),L.T)
#         L = np.dot(Exi_res,np.linalg.inv(Exixit_res))
#         phi = np.diag(np.diagonal(S_res)/n)
#     x_res = []
#     for i in range(n):
#         A = np.linalg.inv(np.eye(2) + np.dot(np.dot(L.T, np.linalg.inv(phi)), L))
#         temp = np.dot(np.dot(A, L.T), np.linalg.inv(phi))
#         x = np.dot(temp, y[i] - mu).reshape(-1, 1)
#         x_res.append(x)
#     x = np.hstack(x_res)
#     return L,phi,x.T


def em(y,iter,L,phi):
    n,q = y.shape
    mu = np.mean(y, axis=0)
    for _ in range(iter):
        Exi_res = 0
        Exixit_res = 0
        S_res = 0
        for i in range(n):
            A = np.linalg.inv(np.eye(2)+ np.dot(np.dot(L.T,np.linalg.inv(phi)),L))
            temp = np.dot(np.dot(A,L.T),np.linalg.inv(phi))
            x = np.dot(temp,y[i]-mu).reshape(-1,1)
            Exi_res += np.dot((y[i]-mu).reshape(-1,1),x.T)
            Exixit_res += np.dot(x,x.T) + A
            S_res += np.dot((y[i]-mu).reshape(-1,1),(y[i]-mu).reshape(-1,1).T)
            S_res -= np.dot(np.dot((y[i]-mu).reshape(-1,1),x.T),L.T)
            S_res -= np.dot(np.dot(L,x),(y[i]-mu).reshape(-1,1).T) - np.dot(np.dot(L,np.dot(x,x.T)+A),L.T)
        L = np.dot(Exi_res,np.linalg.inv(Exixit_res))
        phi = np.diag(np.diagonal(S_res)/n)
    x_res = []
    for i in range(n):
        A = np.linalg.inv(np.eye(2) + np.dot(np.dot(L.T, np.linalg.inv(phi)), L))
        temp = np.dot(np.dot(A, L.T), np.linalg.inv(phi))
        x = np.dot(temp, y[i] - mu).reshape(-1, 1)
        x_res.append(x)
    x = np.hstack(x_res)
    return L,phi,x.T







def ppca(y,iter,L,sigma):
    n,q = y.shape
    mu = np.mean(y, axis=0)
    phi = sigma*np.eye(q)
    for _ in range(iter):
        Exi_res = 0
        Exixit_res = 0
        S_res = 0
        for i in range(n):
            A = np.linalg.inv(np.eye(2)+ np.dot(np.dot(L.T,np.linalg.inv(phi)),L))
            temp = np.dot(np.dot(A,L.T),np.linalg.inv(phi))
            x = np.dot(temp,y[i]-mu).reshape(-1,1)
            Exi_res += np.dot((y[i]-mu).reshape(-1,1),x.T)
            Exixit_res += np.dot(x,x.T) + A
            S_res += np.dot((y[i]-mu).reshape(-1,1),(y[i]-mu).reshape(-1,1).T)
            S_res -= np.dot(np.dot((y[i]-mu).reshape(-1,1),x.T),L.T)
            S_res -= np.dot(np.dot(L,x),(y[i]-mu).reshape(-1,1).T) - np.dot(np.dot(L,np.dot(x,x.T)+A),L.T)
        L = np.dot(Exi_res,np.linalg.inv(Exixit_res))
        sigma = np.mean(np.diagonal(S_res)/n)
        phi = sigma*np.eye(q)
    x_res = []
    for i in range(n):
        A = np.linalg.inv(np.eye(2) + np.dot(np.dot(L.T, np.linalg.inv(phi)), L))
        temp = np.dot(np.dot(A, L.T), np.linalg.inv(phi))
        x = np.dot(temp, y[i] - mu).reshape(-1, 1)
        x_res.append(x)
    x = np.hstack(x_res)
    return L,sigma,x.T

Lambda = np.ones((7, 2))
for i in range(7):
    if i <= 2:
        Lambda[i, 1] = 0
    elif i >= 4:
        Lambda[i, 0] = 0
mean = [1,0,1,0,1,0,1,1,0,1,0,1,0,1]

Sigma = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,3]
res_si = []
phi_init = np.eye(7)

for sigma in tqdm(Sigma):
    L_init = np.random.multivariate_normal(mean, sigma * np.eye(14), size=1).reshape(-1, 2)
    res = []
    for _ in range(20):
        (y,x) = generate()
        y = y.T
        x = x.T
        L,phi,x_1 = em(y,75,L_init,phi_init)
        res.append(np.linalg.norm(x-x_1))
    res_si.append(np.mean(res))

plt.plot(Sigma,res_si,marker="o")
plt.xlabel("sigma square")
plt.ylabel("MSE")
plt.show()

res_col = []
num = []
for i in range(30):
    num.append(i)
    (y, x) = generate()
    y = y.T
    L_init = np.random.random((7,2))
    L, phi, x_1 = em(y, 75, L_init, phi_init)
    res_col.append(np.linalg.norm(np.dot(Lambda,Lambda.T)-np.dot(L,L.T)))

plt.plot(num,res_col,marker="o")
plt.xlabel("number of experiment")
plt.ylabel("F-norm")
plt.show()

(y,x) = generate()
y= y.T
L_init = np.random.multivariate_normal(mean, 0.4 * np.eye(14), size=1).reshape(-1, 2)
L,phi,x_a = em(y,75,L_init,phi_init)
L_init = np.random.random((7,2))
L,phi,x_b = em(y,75,L_init,phi_init)
L,sigma,x_c = ppca(y,75,L_init,phi_init)
plt.scatter(x_a[:,0],x_a[:,1],color = "r",label="Part(a)")
plt.scatter(x_b[:,0],x_b[:,1],color = "b",label="Part(b)")
plt.scatter(x_c[:,0],x_c[:,1],color = "y",label="Part(c)")
plt.legend()
plt.show()
