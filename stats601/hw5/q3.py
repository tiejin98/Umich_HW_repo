import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt



def sigmoid(X):
    return 1/(1+np.exp(-X))

def generator():
    mean = np.zeros(2)
    sigma = np.eye(2)
    Z = np.random.normal(size=1100)
    x = np.random.multivariate_normal(mean,sigma,size=1100)
    a1 = np.array([3,3])
    a2 = np.array([3,-3])
    fea1 = sigmoid(np.dot(x,a1))
    fea2 = np.dot(x,a2)
    y = fea1 + fea2*fea2 +0.3*Z
    return x,y

x,y = generator()
x_train = torch.from_numpy(x[:100]).float()
y_train = torch.from_numpy(y[:100]).float().unsqueeze(-1)
x_test = torch.from_numpy(x[100:]).float()
y_test = torch.from_numpy(y[100:]).float().unsqueeze(-1)

wd_list= [1e-3,1e-2,0.1,0.5,1,5,10]
for wd in wd_list:
    model = nn.Sequential(
        nn.Linear(2,10),
        nn.Linear(10,1)
    )
    loss = nn.MSELoss()
    loss1 = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(),lr=0.001,weight_decay=wd)

    num_epoch = 301
    train_error = []
    test_error = []
    for _ in range(num_epoch):
        y_train_pred = model(x_train)
        train_loss = loss(y_train_pred,y_train)
        train_error.append(train_loss.detach().numpy())
        y_test_pred = model(x_test)
        test_loss = loss1(y_test_pred,y_test)
        test_error.append(test_loss.detach().numpy())
        optim.zero_grad()
        train_loss.backward()
        optim.step()

    plt.plot(list(range(num_epoch)),train_error,color="red",label="train_error")
    plt.plot(list(range(num_epoch)),test_error,color="blue",label="test_error")
    plt.ylabel("mse loss")
    plt.xlabel("number of training epochs")
    plt.title("weight_decay:{}".format(wd))
    plt.legend()
    plt.show()

hidden_unit = list(range(1,11))
train_error = []
test_error = []
for num in hidden_unit:
    print(num)
    model = nn.Sequential(
        nn.Linear(2,num),
        nn.Linear(num,1)
    )
    loss = nn.MSELoss()
    loss1 = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(),lr=0.001,weight_decay=0.1)

    num_epoch = 101
    for _ in range(num_epoch):
        y_train_pred = model(x_train)
        train_loss = loss(y_train_pred,y_train)
        y_test_pred = model(x_test)
        test_loss = loss1(y_test_pred,y_test)
        if _ == 100:
            train_error.append(train_loss.detach().numpy())
            test_error.append(test_loss.detach().numpy())
        optim.zero_grad()
        train_loss.backward()
        optim.step()

plt.plot(hidden_unit,train_error,color="red",label="train_error")
plt.plot(hidden_unit,test_error,color="blue",label="test_error")
plt.ylabel("mse loss after 100 epochs")
plt.xlabel("number of hidden units")
plt.legend()
plt.show()





