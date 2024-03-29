import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,LinearRegression
from pygam import LogisticGAM
from scipy.special import expit
from scipy import optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.optimize import _check_optimize_result
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer



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



def plot_boundary(fit, x_0,x_1, ax, res=100, dim=(0, 1)):
    ax.scatter(x_0[:, dim[0]], x_0[:, dim[1]],marker="o",label = "class 0")
    ax.scatter(x_1[:, dim[0]], x_1[:, dim[1]],marker="x",label = "class 1")
    xrange = ax.get_xlim()
    yrange = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xrange, res),
                         np.linspace(*yrange, res))
    xxf = xx.flatten()
    yyf = yy.flatten()
    # xm = x_0.mean(0)
    # XX = np.vstack([xm for _ in range(xxf.shape[0])])
    XX = np.zeros(shape=(xxf.shape[0],2))
    XX[:, dim[0]] = xxf
    XX[:, dim[1]] = yyf
    ZZ = fit.predict_proba(XX)
    zz = ZZ[:, 1].reshape(xx.shape)
    # ax.pcolormesh(xx, yy, zz, cmap='red_blue_classes', shading="auto",
    #                norm=colors.Normalize(0., 1.), zorder=0)
    ax.contour(xx, yy, zz, [0.5], linewidths=2., colors='black')
    plt.legend()
    plt.show()

def plot_boundary_linear(fit, x_0,x_1, ax, res=100, dim=(0, 1)):
    ax.scatter(x_0[:, dim[0]], x_0[:, dim[1]],marker="o",label = "class 0")
    ax.scatter(x_1[:, dim[0]], x_1[:, dim[1]],marker="x",label = "class 1")
    xrange = ax.get_xlim()
    yrange = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xrange, res),
                         np.linspace(*yrange, res))
    xxf = xx.flatten()
    yyf = yy.flatten()
    # xm = x_0.mean(0)
    # XX = np.vstack([xm for _ in range(xxf.shape[0])])
    XX = np.zeros(shape=(xxf.shape[0],2))
    XX[:, dim[0]] = xxf
    XX[:, dim[1]] = yyf
    ZZ = fit.predict(XX)
    zz = ZZ[:].reshape(xx.shape)
    # ax.pcolormesh(xx, yy, zz, cmap='red_blue_classes', shading="auto",
    #                norm=colors.Normalize(0., 1.), zorder=0)
    ax.contour(xx, yy, zz, [0.5], linewidths=2., colors='black')
    plt.legend()
    plt.show()

def plot_boundary_gam(fit, x_0,x_1, ax, res=100, dim=(0, 1)):
    ax.scatter(x_0[:, dim[0]], x_0[:, dim[1]],marker="o",label = "class 0")
    ax.scatter(x_1[:, dim[0]], x_1[:, dim[1]],marker="x",label = "class 1")
    xrange = ax.get_xlim()
    yrange = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(*xrange, res),
                         np.linspace(*yrange, res))
    xxf = xx.flatten()
    yyf = yy.flatten()
    # xm = x_0.mean(0)
    # XX = np.vstack([xm for _ in range(xxf.shape[0])])
    XX = np.zeros(shape=(xxf.shape[0],2))
    XX[:, dim[0]] = xxf
    XX[:, dim[1]] = yyf
    ZZ = fit.predict_proba(XX)
    zz = ZZ[:].reshape(xx.shape)
    # ax.pcolormesh(xx, yy, zz, cmap='red_blue_classes', shading="auto",
    #                norm=colors.Normalize(0., 1.), zorder=0)
    ax.contour(xx, yy, zz, [0.5], linewidths=2., colors='black')
    plt.legend()
    plt.show()

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


train_data = np.loadtxt("classification_dat.txt",dtype=float,delimiter=" ")
test_data = np.loadtxt("classification_test-1.txt",dtype=float,delimiter=" ")
train_x = train_data[:,:-1]
train_y = train_data[:,-1]
test_x = test_data[:,:-1]
test_y = test_data[:,-1]

x_0 = train_x[train_y == 0]
x_1 = train_x[train_y == 1]
plt.scatter(x_0[:,0],x_0[:,1],marker="o",label = "class 0")
plt.scatter(x_1[:,0],x_1[:,1],marker="x",label = "class 1")
plt.legend()
plt.show()

print(x_0.shape)
print(x_1.shape)
LDA = LinearDiscriminantAnalysis()
LDA.fit(train_x,train_y)
a = plt.subplot()
plot_boundary(LDA,x_0,x_1,a)

logit = LogisticRegression()
logit.fit(train_x,train_y)
a = plt.subplot()
plot_boundary(logit,x_0,x_1,a)

linear = LinearRegression()
linear.fit(train_x,train_y)
a = plt.subplot()
plot_boundary_linear(linear,x_0,x_1,a)

logit_gam = LogisticGAM()
logit_gam.fit(train_x,train_y)
a = plt.subplot()
plot_boundary_gam(logit_gam,x_0,x_1,a)

kernel_logit = KernelLogisticRegression()
kernel_logit.fit(train_x,train_y)
a = plt.subplot()
plot_boundary(kernel_logit,x_0,x_1,a)

y_LDA = LDA.predict(test_x)
y_logit = logit.predict(test_x)
y_linear = linear.predict(test_x)
y_linear[y_linear>=0.5] = 1
y_linear[y_linear<0.5] = 0
y_logit_gam = logit_gam.predict(test_x)
y_logit_kernel = kernel_logit.predict(test_x)

print("LDA test:",test_error_01(test_y,y_LDA))
print("logit test:",test_error_01(test_y,y_logit))
print("linear test:",test_error_01(test_y,y_linear))
print("addtive logit test:",test_error_01(test_y,y_logit_gam))
print("kernel logit test:",test_error_01(test_y,y_logit_kernel))