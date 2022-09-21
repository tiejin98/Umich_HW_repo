import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

train_data = np.loadtxt("spam-train.txt",dtype=float,delimiter=",")
test_data = np.loadtxt("spam-test.txt",dtype=float,delimiter=",")

train_x = train_data[:,:-1]
train_y = train_data[:,-1]
test_x = test_data[:,:-1]
test_y = test_data[:,-1]

def standardize(x):
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    return [(x-mean)/std,mean,std]

def standardize_with_mean(x,mean,std):
    return (x-mean)/std

def positive_log(x):
    return np.log(1+x)

def indicator(x):
    data = deepcopy(x)
    data[data > 0] = 1
    data[data < 0] = 0
    return data

stand_train_x,mean,std = standardize(train_x)
stand_test_x = standardize_with_mean(test_x,mean,std)


class logit_regrssion():
    def __init__(self,x,y,lamb=1,bias=False):
        """
        :param x: x is the data, it should be n * d matrix
        :param y: y is the label.
        :param lamb: the parameter control the penalty
        :param bias: control whether the model have the constant item.
        """
        self.bias = bias
        self.x = x
        self.y = y
        if bias:
            self.theta = np.zeros(self.x.shape[1]+1)
        else:
            self.theta = np.zeros(self.x.shape[1])
        self.lamb = lamb

    def forward(self,x,label=True):
        temp_x = deepcopy(x)
        if self.bias == True:
            b = np.ones(x.shape[0])
            temp_x = np.insert(temp_x,0,b,axis=1)
        temp_y = 1/(1+np.exp(-1*(np.dot(temp_x,self.theta))))
        if label == True:
            res = []
            for y_bar in temp_y:
                if y_bar >= 0.5:
                    res.append(1)
                else:
                    res.append(0)
            return res
        else:
            return temp_y

    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))

    def test_error_01(self,test_y,y_bar):
        test_y = np.array(test_y).reshape(-1,)
        y_bar = np.array(y_bar).reshape(-1,)
        if len(test_y) != len(y_bar):
            raise ValueError('two y should have same length')
        total = 0
        for i in range(len(y_bar)):
            if y_bar[i] - test_y[i] != 0:
                total += 1
        return total/len(y_bar)

    def likelihood_function(self,x,y):
        forward_res = self.forward(x,label=False)
        L = -np.dot(y,np.log(forward_res+1e-7)) - np.dot(1-y,np.log(1-forward_res+1e-7)) \
            + self.lamb*np.dot(self.theta.T,self.theta)
        return L

    def first_order(self):
        temp_x = deepcopy(self.x)
        if self.bias:
            b = np.ones(temp_x.shape[0])
            temp_x = np.insert(temp_x,0,b,axis=1)
        else:
            temp_x = self.x
        forward_res = self.forward(self.x,label=False)
        return np.dot(self.y-forward_res,temp_x)

    def second_order(self):
        if self.bias:
            b = np.ones(self.x.shape[0])
            temp_x = np.insert(self.x,0,b,axis=1)
        else:
            temp_x = self.x
        dii = -np.exp(-1*(np.dot(temp_x,self.theta)))/(1+np.exp(-1*(np.dot(temp_x,self.theta))))**2
        D = np.diag(dii)
        temp = np.dot(temp_x.T,D)
        return np.dot(temp,temp_x) - self.lamb*np.eye(self.theta.shape[0])

    def one_step_newton(self,lr=1):
        First = self.first_order()
        H = self.second_order()
        self.theta -= lr*np.dot(np.linalg.inv(H),First)
        # temp_x = deepcopy(self.x)
        # if self.bias:
        #     b = np.ones(self.x.shape[0])
        #     temp_x = np.insert(temp_x, 0, b, axis=1)
        # else:
        #     temp_x = self.x
        # N, d = temp_x.shape
        # preds = self.sigmoid(temp_x@self.theta)
        # grad = temp_x.T@(self.y - preds)
        # H = - temp_x.T @ (np.expand_dims((preds * (1 - preds)), -1) * temp_x) - self.lamb*np.eye(d)
        # self.theta -= np.matmul(np.linalg.inv(H), grad)


    def fit_epoch(self,lr=1,epoch=10):
        print("Before training, we have log-likelihood is {}".format(self.likelihood_function(self.x,self.y)))
        for _ in range(epoch):
            self.one_step_newton(lr)
        print("After {} round, we can get log-likelihood is {}".format(epoch,self.likelihood_function(self.x,self.y)))

    def fit(self,test_x=None,test_y=None,lr=1,eps=1e-4):
        i = 0
        print("Before Newton, the trianing error(0-1) is: {J}".format(J=self.test_error_01(self.y,self.forward(self.x))))

        while True:
            theta_old = deepcopy(self.theta)
            i += 1
            self.one_step_newton(lr)
            # if i % 10 == 0:
            #     if type(test_x) != type(None) and type(test_y) != type(None):
            #         print("In {I} iterations, the training error(0-1) is: {J}, the test error(0-1) is:{T}".
            #             format(I=i,J=self.test_error_01(self.y,self.forward(self.x)),
            #                    T=self.test_error_01(test_y,self.forward(test_x))))
            #     else:
            #         print("In {I} iterations, the objective function is: {J},".format(I=i,J=J1))
            if np.sum(abs(theta_old-self.theta)) < eps:
                break
        if type(test_x) != type(None) and type(test_y) != type(None):
            print("After {I} iterations, it converges, the training error is: {J},the test error(0-1) is:{T}".
                  format(I=i, J=self.test_error_01(self.y,self.forward(self.x)),T=self.test_error_01(test_y,self.forward(test_x))))

class LDA():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.pi = np.sum(y)/y.shape[0]
        sigma1 = np.cov(x[y==0].T)
        sigma2 = np.cov(x[y==1].T)
        self.sigma = (sigma1*(x[y==0].shape[0]) + sigma2*(x[y==1].shape[0]))/(x.shape[0]-2)
        x_0 = x[y==0]
        x_1 = x[y==1]
        self.mu0 = np.mean(x_0,axis=0)
        self.mu1 = np.mean(x_1,axis=0)
        diff = self.mu1-self.mu0
        plus = self.mu1+self.mu0
        sigma_inv = np.linalg.inv(self.sigma)
        self.beta = np.dot(sigma_inv,(self.mu1-self.mu0).T)
        temp = np.dot(diff,sigma_inv)
        self.gamma = -0.5*np.dot(temp,plus.T) + np.log((1-self.pi)/self.pi)

    def predict(self,x):
        res = np.dot(x,self.beta)+self.gamma
        res[res>=0] = 1
        res[res<0] = 0
        return res

    def test_error_01(self,test_y,y_bar):
        test_y = np.array(test_y).reshape(-1,)
        y_bar = np.array(y_bar).reshape(-1,)
        if len(test_y) != len(y_bar):
            raise ValueError('two y should have same length')
        total = 0
        for i in range(len(y_bar)):
            if y_bar[i] - test_y[i] != 0:
                total += 1
        return total/len(y_bar)



def nb_train(matrix, category):
    # Implement your algorithm and return
    state = {}
    def estimate_pi_k(x, y, k):
        x_k = x[np.where(y == k)]
        return np.log(x_k.shape[0] / x.shape[0])

    def estimate_log_pkj(x, y, k, a=1):
        res = []
        x_k = x[np.where(y == k)]
        n_k = np.sum(x_k)
        for j in range(x_k.shape[1]):
            n_kj = np.sum(x_k[:, j])
            p_kj = (n_kj + a) / (n_k + a * x_k.shape[1])
            res.append(np.log(p_kj))
        return res
    k_list = [0,1]
    state['pi'] = [estimate_pi_k(matrix,category,k) for k in k_list]
    state['pkj'] = [estimate_log_pkj(matrix,category,k) for k in k_list]
    return state

def nb_test(matrix, state):
    # Classify each email in the test set (each row of the document matrix) as 1 for SPAM and 0 for NON-SPAM
    pi = state['pi']
    pkj = state['pkj']
    temp_res = []
    for i in range(len(pi)):
        temp_res.append(np.dot(matrix,np.array(pkj[i]).reshape(-1,1))+pi[i])
    temp_res = np.array(temp_res)
    return np.argmax(temp_res,axis=0).reshape(-1,)

from scipy.special import expit
from scipy import optimize
from matplotlib import colors

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.optimize import _check_optimize_result
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer


cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)

def plot_boundary(fit, X, y, ax, res=100, dim=(0, 1)):
    ax.scatter(X[:, dim[0]], X[:, dim[1]], c=y, cmap='red_blue_classes',
              linewidths=1., edgecolors="white")
    xrange = ax.get_xlim()
    yrange = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xrange, res),
                         np.linspace(*yrange, res))
    xxf = xx.flatten()
    yyf = yy.flatten()
    xm = X.mean(0)
    XX = np.vstack([xm for _ in range(xxf.shape[0])])
    XX[:, dim[0]] = xxf
    XX[:, dim[1]] = yyf
    ZZ = fit.predict_proba(XX)
    zz = ZZ[:, 1].reshape(xx.shape)
    ax.pcolormesh(xx, yy, zz, cmap='red_blue_classes', shading="auto",
                   norm=colors.Normalize(0., 1.), zorder=0)
    ax.contour(xx, yy, zz, [0.5], linewidths=2., colors='white')


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

def _loss_and_grad(w, K, y, alpha, clip=30):
    """
    Computes the loss and the gradient
    The loss is the negative likelihood function
    :param w:
        array-like oh shape (n_samples,)
        weights
    :param K:
        array-like of shape (n_sample, n_samples)
        the kernel matrix
    :param y:
        array-like of shape (n_samples,)
        labels
    :param alpha:
        float
        penalty parameter
    :return:
        out : float
            The loss
        grad : ndarray of shape (X.shape[0],)
            The gradient
    """

    n_samples = K.shape[0]

    linear_prediction = K.dot(w)
    penalty = (alpha / 2.) * w.T.dot(K).dot(w)

    # Loss for kernel logistic regression is the negative likelihood
    out = np.sum(-y * linear_prediction + np.log(1 + np.exp(linear_prediction))) + penalty

    z = expit(linear_prediction)
    z0 = y - z - alpha * w

    grad = -K.dot(z0)

    return out, grad


def _kernel_logistic_regression_path(K, y, max_iter, tol=1e-4, coef=None,
                                     solver='lbfgs', check_input=True,
                                     C = 1):
    """
    Compute the kernel logistic regression model
    :param K:
        array-like of shape (n_sample, n_features)
        Input pitchfx
    :param y:
        array-like of shape (n_samples,)
        Input pitchfx, target values
    :param tol:
        float, default = 1e-4
        The stopping criterion for the solver
    :param coef:
        array-like of shape (n_samples,)
        Initialisation values of coefficients for the regression
    :param solver:
        str
        The solver to be used
    :param check_input:
        bool, default = True
        Determines whether the input pitchfx should be checked
    :return:
        w0 : ndarray of shape
    """

    # TODO: implement
    # if check_input:

    n_samples, n_features = K.shape
    classes = np.unique(y)

    if not classes.size == 2:
        raise ValueError("Only binary Classification.")

    func = _loss_and_grad

    if coef is None:
        w0 = np.zeros(n_samples, order='F', dtype=K.dtype)
    else:
        w0 = coef

    # TODO: implement other solvers
    if solver == 'lbfgs':
        iprint = [-1, 50, 1, 100, 101]
        opt_res = optimize.minimize(
            func, w0, method="L-BFGS-B", jac=True,
            args=(K, y, 1. / C, 30),
            options={"iprint": iprint, "gtol": tol, "maxiter": max_iter}
        )

    n_iter = _check_optimize_result(solver, opt_res, max_iter)

    w0, loss = opt_res.x, opt_res.fun

    return np.array(w0), n_iter


class KernelLogisticRegression(BaseEstimator, ClassifierMixin):
    """ Binary Classifier using Kernel Logistic Regression
    ----------
    kernel : str, default='rbf_kernel'
        Used to determine which kernel function is to be used when generating
        the kernel matrix
    learning_rate : float, default=1
        The learning rate for gradient descent
    gamma : float, default = 1
        Used during the creation for the Gaussian kernel matrix
    C : float, default = 1
        The inverse of the penalty parameter
    """

    def __init__(self,
                 kernel='rbf',
                 learning_rate=1,
                 gamma=1,
                 degree=3,
                 coef0=1,
                 C=1,
                 tol=1e-4,
                 kernel_params=None,
                 max_iter=1000):
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.C = C
        self.tol = tol
        self.kernel_params = kernel_params
        self.max_iter = max_iter

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        if self.C < 0:
            raise ValueError("Penalty must be positive")

        # Necessary for prediction
        self.X_ = X

        X, y = check_X_y(X, y, accept_sparse=True)
        self.label_encoder_ = LabelBinarizer(neg_label=0, pos_label=1)
        y_ = self.label_encoder_.fit_transform(y).reshape((-1))

        self.classes_ = self.label_encoder_.classes_
        K = self._get_kernel(X)

        self.coef_, self.n_iter_ = _kernel_logistic_regression_path(K, y_, tol=self.tol, coef=None,
                                     C=self.C, solver='lbfgs', check_input=True,
                                     max_iter=self.max_iter)

        self.is_fitted_ = True

        return self

    def decision_function(self, X):

        check_is_fitted(self, ["X_", "coef_"])

        K = self._get_kernel(X, self.X_)

        scores = K.dot(self.coef_)

        return scores

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        scores = self.decision_function(X)

        indices = (scores > 0).astype(int)

        return self.classes_[indices]

    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """

        check_is_fitted(self)

        pred_1 = expit(self.decision_function(X).clip(-30, 30)).reshape((-1, 1))

        return np.hstack((1.0 - pred_1, pred_1))

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """

        check_is_fitted(self)

        return np.log(self.predict_proba(X))

model = logit_regrssion(stand_train_x,train_y,lamb=0.05,bias=True)
model.fit(standardize_with_mean(test_x,mean,std),test_y)
model = logit_regrssion(positive_log(train_x),train_y,lamb=0.05,bias=True)
model.fit(positive_log(test_x),test_y)
model = logit_regrssion(indicator(train_x),train_y,lamb=0.05,bias=True)
model.fit(indicator(test_x),test_y)

model = LDA(stand_train_x,train_y)
pred = model.predict(standardize_with_mean(test_x,mean,std))
print(model.test_error_01(test_y,pred))


model = LDA(positive_log(train_x),train_y)
pred = model.predict(positive_log(test_x))
print(model.test_error_01(test_y,pred))

state = nb_train(indicator(train_x),train_y)
pred = nb_test(indicator(test_x),state)
print(model.test_error_01(test_y,pred))



C = [0.01,0.05,0.1,0.3,0.5,0.7,1.0,1.5,2.0,2.5,3.0,4.0]
gamma = [0.01,0.05,0.1,0.3,0.5,0.7,1.0,1.5,2.0,2.5,3.0,4.0]
degree = [1,2,3,4,5,6,7,8,9,10,11,12]
c_res_train = []
c_res_test = []
g_res_train = []
g_res_test = []
# for i in range(len(C)):
#     klr = KernelLogisticRegression(C=C[i])
#     klr.fit(stand_train_x,train_y)
#     pred = klr.predict(standardize_with_mean(test_x,mean,std))
#     c_res_test.append(test_error_01(test_y,pred))
#     y_bar = klr.predict(stand_train_x)
#     c_res_train.append(test_error_01(train_y,y_bar))
#     a = plt.subplot(4,3,i+1)
#     #a = plt.subplot(1,1,1)
#     plot_boundary(klr,standardize_with_mean(test_x,mean,std),test_y,a)
# plt.show()
#
# plt.plot(C,c_res_train,color="red",label = "train set 0-1 error")
# plt.plot(C,c_res_test,color="blue",label = "test set 0-1 error")
# plt.ylabel('standard 0-1 error')
# plt.xlabel("C")
# plt.legend()
# plt.show()
#


#
# for i in range(len(gamma)):
#     klr = KernelLogisticRegression(gamma=gamma[i])
#     klr.fit(stand_train_x,train_y)
#     pred = klr.predict(standardize_with_mean(test_x,mean,std))
#     g_res_test.append(test_error_01(test_y,pred))
#     y_bar = klr.predict(stand_train_x)
#     g_res_train.append(test_error_01(train_y,y_bar))
#     a = plt.subplot(4,3,i+1)
#     #a = plt.subplot(1,1,1)
#     plot_boundary(klr,standardize_with_mean(test_x,mean,std),test_y,a)
#
# plt.show()
# plt.plot(gamma,g_res_train,color="red",label = "train set 0-1 error")
# plt.plot(gamma,g_res_test,color="blue",label = "test set 0-1 error")
# plt.ylabel('standardize 0-1 error')
# plt.xlabel("gamma")
# plt.legend()
# plt.show()
#

# for i in range(len(gamma)):
#     klr = KernelLogisticRegression(gamma=gamma[i])
#     klr.fit(positive_log(train_x),train_y)
#     pred = klr.predict(positive_log(test_x))
#     c_res_test.append(test_error_01(test_y,pred))
#     y_bar = klr.predict(positive_log(train_x))
#     c_res_train.append(test_error_01(train_y,y_bar))
#     a = plt.subplot(4,3,i+1)
#     #a = plt.subplot(1,1,1)
#     plot_boundary(klr,positive_log(test_x),test_y,a)
# plt.show()
#
# plt.plot(C,c_res_train,color="red",label = "train set 0-1 error")
# plt.plot(C,c_res_test,color="blue",label = "test set 0-1 error")
# plt.ylabel('log-transform 0-1 error')
# plt.xlabel("gamma")
# plt.legend()
# plt.show()

# for i in range(len(degree)):
#     klr = KernelLogisticRegression(kernel="poly",degree=degree[i],gamma=1/58)
#     klr.fit(positive_log(train_x),train_y)
#     pred = klr.predict(positive_log(test_x))
#     c_res_test.append(test_error_01(test_y,pred))
#     y_bar = klr.predict(positive_log(train_x))
#     c_res_train.append(test_error_01(train_y,y_bar))
#     a = plt.subplot(4,3,i+1)
#     #a = plt.subplot(1,1,1)
#     plot_boundary(klr,positive_log(test_x),test_y,a)
# plt.show()
#
# plt.plot(degree,c_res_train,color="red",label = "train set 0-1 error")
# plt.plot(degree,c_res_test,color="blue",label = "test set 0-1 error")
# plt.ylabel('log-transfrom 0-1 error')
# plt.xlabel("degree")
# plt.legend()
# plt.show()

# from sklearn.model_selection import GridSearchCV
# params = {
#     "degree": [1,2,3,4],
#     "gamma": [0.001, 0.005,0.01,1/57,0.05, 0.1, 0.5]
# }
#
# cv = GridSearchCV(KernelLogisticRegression(kernel="poly"), params, cv=5, scoring="accuracy").fit(positive_log(train_x), train_y)
# model = cv.best_estimator_
# pred = model.predict(positive_log(test_x))
# print(test_error_01(test_y,pred))