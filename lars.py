import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def min_plus(x) -> float:
    """Minimum over positive components"""
    x = x[x > 0]
    if len(x) == 0:
        return np.inf
    return np.min(x)


def float_comparison(a, b, eps=1e-6) -> bool:
    return np.abs(a - b) < eps


def lin_regression(X: np.array, y: np.array) -> np.array:
    """Beta coefficients for linear regression model: X * B = y + epsilon"""
    return np.linalg.inv(X.T @ X) @ X.T @ y


class Lars():
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
                    min_plus((max(c) + c[A_c]).reshape(-1) / (A_a + a[A_c]).reshape(-1)),
                    min_plus((max(c) - c[A_c]).reshape(-1) / (A_a - a[A_c]).reshape(-1)),
                )
            if not np.isfinite(gamma):
                break
            mi = mi + gamma * u_a
            if float_comparison(gamma, 0):
                # if weights are no longer updated
                break
            path = np.concatenate([path, lin_regression(X, mi)], axis=1)
        return path

    def plot_lars_path(self, path: np.array = None, title="Lars visualization", **fig_kwars):
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

    @staticmethod
    def interpolate_path(path, C):
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

    @staticmethod
    def cut_path(path, C):
        # boundary case
        if C < 0:
            return path[:, 0]
        l1 = np.abs(path).sum(axis=0)  # vector of l1 norm for each step
        # number of the last iteration for which the l1 norm didn't exceed the C constraint
        last_ind = np.max(np.where(l1 < C))
        return path[:, last_ind].reshape((-1, 1))

    def lars_predict(self, X, constraint, interpolate=True) -> np.array:
        X = self.xscaler.transform(X)
        if interpolate:
            coeff = self.interpolate_path(self.path, constraint)
        else:
            coeff = self.cut_path(self.path, constraint)
        y = X @ coeff
        y = self.yscaler.inverse_transform(y)
        return y


def lars_irls(X: np.array, y: np.array) -> np.array:
    """Performs l1 regularized logistic regression based on combination of IRLS and LARS algorithms.
    :param X: matrix of predictors
    :param y: column target vector.
    """
    pass


if __name__ == "__main__":
    from sklearn.datasets import load_boston, load_diabetes

    X, y = load_boston(return_X_y=True)
    bost_custom_lars = Lars(X, y)
    X, y = load_diabetes(return_X_y=True)
    diab_custom_lars = Lars(X, y)
    bost_custom_lars.plot_lars_path(title="Lars for boston", figsize=(8, 6))
    diab_custom_lars.plot_lars_path(title="Lars for diabetes", figsize=(8, 6))
