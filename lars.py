import numpy as np


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


def lars(X: np.array, y: np.array, max_it: int = 1e3) -> np.array:
    """Returns lars path. Each column represents coefficients at each iteration.
    There is i non-zero coefficients at ith column.
    Based on https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-2/Least-angle-regression/10.1214/009053604000000067.full
    :param X: matrix of predictors
    :param y: column target vector. Will be converted to have mean=0 and std=1.
    :returns LARS path"""
    X = np.array(X)
    y = np.array(y)
    y = (y - y.mean()) / y.std()
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
        if float_comparison(gamma,0):
            # if weights are no longer updated
            break
        path = np.concatenate([path, lin_regression(X, mi)], axis=1)
    return path


if __name__ == '__main__':
    from sklearn.datasets import load_boston, load_diabetes
    from sklearn.linear_model import lars_path
    X, y = load_boston(return_X_y=True)
    y = (y - y.mean()) / y.std()
    bost_custom_lars = lars(X, y)
    bost_sklearn_lars = lars_path(X,y)[2]
    X, y = load_diabetes(return_X_y=True)
    y = (y - y.mean()) / y.std()
    diab_custom_lars = lars(X, y)
    diab_sklearn_lars = lars_path(X,y)[2]