import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

def generator():
    mean = np.zeros(10)
    sigma = np.eye(10)
    x = np.random.multivariate_normal(mean,sigma,size=12000)
    sum_x = np.sum(x**2,axis=1)
    y = np.ones(12000)
    y[sum_x<=9.34] = -1

    return x,y



def generator_d():
    mean = np.zeros(10)
    sigma = np.eye(10)
    x1 = np.random.multivariate_normal(mean,sigma,size=6000)
    x2 = np.random.multivariate_normal(mean,sigma,size=24000)
    sum_x = np.sum(x2**2,axis=1)
    x2 = x2[sum_x>12]
    x = np.vstack((x1,x2))
    y = np.ones(12000)
    y[6000:] = -1
    x = x[:12000]
    index = list(range(12000))
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    return x,y

def test_error_01(test_y, y_bar):
    test_y = np.array(test_y).reshape(-1, )
    y_bar = np.array(y_bar).reshape(-1, )
    if len(test_y) != len(y_bar):
        raise ValueError('two y should have same length')
    total = 0
    for i in range(len(y_bar)):
        if y_bar[i] - test_y[i] != 0:
            total += 1
    return total / len(y_bar)

# x,y = generator()
# train_x = x[:2000]
# train_y = y[:2000]
# test_x = x[2000:]
# test_y = y[2000:]
#
# num_iterations = 1000
# test_errors = []
# train_errors = []
# for i in tqdm(range(num_iterations)):
#     model = AdaBoostClassifier(n_estimators=i+1, random_state=42)
#     model.fit(train_x, train_y)
#     y_pred = model.predict(train_x)
#     train_error = test_error_01(train_y,y_pred)
#     train_errors.append(train_error)
#     y_pred = model.predict(test_x)
#     test_error = test_error_01(test_y,y_pred)
#     test_errors.append(test_error)
#
# plt.plot(list(range(num_iterations)),train_errors,color="red",label="train_error")
# plt.plot(list(range(num_iterations)),test_errors,color="blue",label="test_error")
# plt.legend()
# plt.show()


x,y = generator_d()
train_x = x[:2000]
train_y = y[:2000]
test_x = x[2000:]
test_y = y[2000:]

num_iterations = 1000
test_errors = []
train_errors = []
for i in tqdm(range(num_iterations)):
    model = AdaBoostClassifier(n_estimators=i+1, random_state=42)
    model.fit(train_x, train_y)
    y_pred = model.predict(train_x)
    train_error = test_error_01(train_y,y_pred)
    train_errors.append(train_error)
    y_pred = model.predict(test_x)
    test_error = test_error_01(test_y,y_pred)
    test_errors.append(test_error)

plt.plot(list(range(num_iterations)),train_errors,color="red",label="train_error")
plt.plot(list(range(num_iterations)),test_errors,color="blue",label="test_error")
plt.legend()
plt.show()
