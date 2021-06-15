import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import line_search
import pandas as pd


def min_plus(x) -> float:
    """Minimum over positive components"""
    x = x[x > 0]
    if len(x) == 0:
        return np.inf
    return np.min(x)


def float_comparison(a, b, eps=1e-6) -> bool:
    return np.abs(a - b) < eps


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def sigmoid_prim(x):
    s = sigmoid(x)
    return s * (1 - s)


def lin_regression(X: np.array, y: np.array) -> np.array:
    """Beta coefficients for linear regression model: X * B = y + epsilon"""
    return np.linalg.inv(X.T @ X) @ X.T @ y


class Lars:
    def __init__(self, X: np.array, y: np.array):
        self.xscaler = StandardScaler()
        self.X = self.xscaler.fit_transform(X)
        self.yscaler = StandardScaler()
        y = y.reshape((-1, 1))
        self.y = self.yscaler.fit_transform(y)
        self.path = self.lars(self.X, self.y)

    @staticmethod
    def lars(X: np.array, y: np.array) -> np.array:
        """Returns lars path. Each column represents coefficients at each iteration.
        There is i non-zero coefficients at ith column.
        Based on https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-2/Least-angle-regression/10.1214/009053604000000067.full
        :param X: matrix of predictors. Each predictor should be converted to have mean=0 and std=1.
        :param y: column target vector. Should be converted to have mean=0 and std=1.
        :returns LARS path"""
        X = np.array(X)
        y = np.array(y)
        y = y.reshape((-1, 1))
        mi = np.zeros((y.shape[0], 1))
        path = np.zeros((X.shape[1], 1))
        while True:
            c = np.dot(X.T, y - mi)
            A = np.arange(len(c)).reshape((-1, 1))[
                float_comparison(np.abs(c), np.max(np.abs(c)))
            ]
            s = np.sign(c)
            X_a = s[A].T * X[:, A]
            G_a = np.dot(X_a.T, X_a)
            A_a = np.sum(np.linalg.inv(G_a)) ** (-0.5)
            u_a = A_a * np.dot(X_a, np.linalg.inv(G_a).sum(axis=1)[:, None])
            a = np.dot(X.T, u_a)
            A_c = np.arange(len(c)).reshape((-1, 1))[
                False == float_comparison(np.abs(c), np.max(np.abs(c)))
                ]
            if len(A_c) == 0:
                gamma = max(c) / A_a
            else:
                gamma = np.minimum(
                    min_plus(
                        (max(c) + c[A_c]).reshape(-1) / (A_a + a[A_c]).reshape(-1)
                    ),
                    min_plus(
                        (max(c) - c[A_c]).reshape(-1) / (A_a - a[A_c]).reshape(-1)
                    ),
                )
            if not np.isfinite(gamma):
                break
            mi = mi + gamma * u_a
            if float_comparison(gamma, 0):
                # if weights are no longer updated
                break
            path = np.concatenate([path, lin_regression(X, mi)], axis=1)
        return path

    def plot_lars_path(
            self, path: np.array = None, title="Lars visualization", **fig_kwars
    ):
        if path is None:
            path = self.path
        plt.figure(**fig_kwars)
        l1 = np.abs(path).sum(axis=0)
        for i in range(path.shape[0]):
            plt.plot(l1, path[i, :])
        plt.xlabel("L1 norm of all variables")
        plt.ylabel("Value of each variable")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def interpolate_path(self, C, path: np.array = None):
        if path is None:
            path = self.path
        # boundary cases
        if C < 0:
            return path[:, 0]
        l1 = np.abs(path).sum(axis=0)  # vector of l1 for each step
        if C > l1[-1]:
            return path[:, -1]
        # number of the last iteration for which the l1 norm didn't exceed the C constraint
        last_ind = np.max(np.where(l1 < C))
        # the interpolation can be written as a convex combination of the last and last+1 step
        # lambda * step_n + (1-lambda) * step_n+1
        # below we compute the lambda in convex combination
        lambd = (C - l1[last_ind]) / (l1[last_ind + 1] - l1[last_ind])
        approx = lambd * path[:, last_ind + 1] + (1 - lambd) * path[:, last_ind]
        return approx.reshape((-1, 1))

    def cut_path(self, C, path: np.array = None):
        if path is None:
            path = self.path
        # boundary case
        if C < 0:
            return path[:, 0]
        l1 = np.abs(path).sum(axis=0)  # vector of l1 norm for each step
        # number of the last iteration for which the l1 norm didn't exceed the C constraint
        last_ind = np.max(np.where(l1 < C))
        return path[:, last_ind].reshape((-1, 1))

    def predict(self, X, constraint, interpolate=True) -> np.array:
        X = self.xscaler.transform(X)
        if interpolate:
            coeff = self.interpolate_path(self.path, constraint)
        else:
            coeff = self.cut_path(self.path, constraint)
        y = X @ coeff
        y = self.yscaler.inverse_transform(y)
        return y


def lars_irls(X, y, contraint: float, max_iter=1e2, alpha=0.5) -> np.array:
    """Performs l1 regularized logistic regression based on combination of IRLS and LARS algorithms.
    Based on https://www.researchgate.net/publication/220269312_Efficient_L1_Regularized_Logistic_Regression
    :param X: matrix of predictors
    :param y: column target vector.
    :returns: vector of coefficients
    """
    theta = np.zeros((X.shape[1], 1))
    y = y.reshape((-1, 1))
    for k in range(int(max_iter)):
        sigm = sigmoid(X @ theta)
        Lambda = np.diag((sigm * (1 - sigm)).ravel())
        z = (X @ theta).reshape((-1, 1)) + np.array(
            [(1 - sigmoid(y[i, 0] * theta.T @ X[i, :])) * y[i, 0] / Lambda[i, i] for i in range(X.shape[0])]).reshape(
            (-1, 1))
        gamma = Lars((Lambda ** 0.5) @ X, (Lambda ** 0.5) @ z).interpolate_path(contraint)
        theta = (1 - alpha) * theta + alpha * gamma.reshape(
            (-1, 1))
    return theta


if __name__ == "__main__":
    from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer

    REGRESSION = True
    if not REGRESSION:
        X, y = load_boston(return_X_y=True)
        bost_lars = Lars(X, y)
        bost_lars.plot_lars_path(title="Lars for boston", figsize=(8, 6))
        X, y = load_diabetes(return_X_y=True)
        diab_lars = Lars(X, y)
        diab_lars.plot_lars_path(title="Lars for diabetes", figsize=(8, 6))
    else:
        X, y = load_breast_cancer(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y)


        def assess_regression(c):
            th = lars_irls(X_train, y_train, c)
            th = np.round(th, 10) + 1e-15
            pred = sigmoid(X_test @ th) > 0.5  # this is the same as sigmoid(X@th)>0.5
            return {'accuracy': (pred.ravel() == y_test.ravel()).mean(),
                    'l1': abs(th).mean(),
                    'n_features': (abs(th) > 1e-15).sum()}


        assessment = pd.DataFrame([assess_regression(c) for c in 10.0 ** np.arange(-16, 6, 2)])

        fig, ax1 = plt.subplots()
        color = "tab:red"
        ax1.set_xlabel("C")
        ax1.set_xscale("log")
        ax1.plot((assessment["l1"]), assessment["accuracy"], color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.set_ylim((0, 1))
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_xscale("log")
        color = "tab:blue"
        ax2.set_ylabel("Number of features", color=color)
        ax1.set_ylabel("Accuracy", color="tab:red")  # we already handled the x-label with ax1
        ax2.plot((assessment["l1"]), assessment["n_features"], color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_ylim((0, 30))
        plt.title("Lasso logistic regression for breast cancer dataset.")
        plt.show()
