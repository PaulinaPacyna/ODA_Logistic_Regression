import numpy as np
import pandas as pd
from copy import copy
from sklearn.preprocessing import MinMaxScaler

# Following functions can be placed in class LogisticModel


def sigmoid(x):
    """
    Numerically stable version of sigmoid (no overflow warnings).
    :param x: array of arguments
    :return: sigmoid of x
    """
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def predict_probabilities(beta, x):
    z = np.dot(x, beta).reshape(x.shape[0], 1)
    return np.apply_along_axis(sigmoid, axis=1, arr=z)


def cost_function(beta, x, y):
    """
    Cross-entropy/log-likelihood cost function (for GD and SGD)
    :param beta:
    :param x:
    :param y:
    :return:
    """
    m = x.shape[0]
    h = predict_probabilities(beta, x)
    return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))


def accuracy(y: np.array, prediction: np.array) -> float:
    """
    Given two vectors consisting of 0 and 1, returns accuracy.
    :param y: true values (0 and 1)
    :param prediction: prediciton (0 and 1)
    :return: accuracy
    """
    return np.mean(y == prediction)


def precision(y: np.array, prediction: np.array) -> float:
    """
    Given two vectors consisting of 0 and 1, returns precision.
    We assume that 1 represents positive value.
    :param y: true values (0 and 1)
    :param prediction: prediciton (0 and 1)
    :return: precision
    """
    true_positives = float(np.sum((y == 1) & (prediction == 1)))
    positives = float(np.sum(prediction))
    return 0 if positives == 0 else true_positives / positives


def recall(y: np.array, prediction: np.array) -> float:
    """
    Given two vectors consisting of 0 and 1, returns recall.
    We assume that 1 represents positive value.
    :param y: true values (0 and 1)
    :param prediction: prediciton (0 and 1)
    :return: recall
    """
    true_positives = float(np.sum((y == 1) & (prediction == 1)))
    true_values = float(np.sum(y))
    return 0 if true_values == 0 else true_positives / true_values


def F_measure(y: np.array, prediction: np.array) -> float:
    """
    Given two vectors consisting of 0 and 1, returns the F measure.
    We assume that 1 represents positive value.
    :param y: true values (0 and 1)
    :param prediction: prediciton (0 and 1)
    :return: recall
    """
    r = recall(y, prediction)
    p = precision(y, prediction)
    return 0 if r + p == 0 else 2 * r * p / (r + p)


class LogisticModel:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, random_state: int = None):
        self.var_names = X.columns
        self.scaler = MinMaxScaler()
        X = self.scaler.fit_transform(X)
        # we save only the scaler of X as an attribute, the scaler of y won't be used anymore
        # it is need only once (for example for transforming y with classess 1,2 to 0,1)
        y = MinMaxScaler().fit_transform(y)
        self.X = np.c_[np.ones((X.shape[0], 1)), np.array(X)]  # adding bias
        self.y = np.array(y)
        self.seed = None if random_state is None else int(random_state)
        self.weights = np.zeros((self.X.shape[1], 1))

    def fit(self, X: pd.DataFrame, return_probabilities: bool = False) -> pd.DataFrame:
        """
        Returns class to which t
        :param X: input dataframe
        :param return_probabilities: should the model return probabilities of belonging to class 1?
        :return: vector of 1 and 0, which represents predicted classes
        """
        X = self.scaler.transform(X)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        if return_probabilities:
            return predict_probabilities(self.weights, X)
        else:
            return np.apply_along_axis(
                np.round, arr=predict_probabilities(self.weights, X), axis=1
            )

    def IRLS(self, n_epochs=100, eps=None, print_progress=False):
        # based on chapter 4.3.3 from Bishop - Pattern Recognition And Machine Learning
        if not (eps is None):
            prev_weights = copy(self.weights)

        y = self.y.reshape(self.y.size, 1)
        for i in range(n_epochs):
            y_ = predict_probabilities(self.weights, self.X)
            R = np.identity(y_.size)
            R = R * (y_ * (1 - y_))
            delta = np.linalg.pinv(self.X.T.dot(R).dot(self.X)).dot(
                self.X.T.dot(y_ - y)
            )
            self.weights -= delta

            if print_progress:
                print(f"Number of epoch: {i + 1}/{n_epochs}")

            if not (eps is None):
                if np.linalg.norm(self.weights - prev_weights) < eps:
                    if print_progress:
                        print(f"Algorithm converged after {i + 1} iterations")
                    break
                prev_weights = copy(self.weights)

    def GD(self, n_epochs=100, learning_rate=0.05, eps=None):
        if not (eps is None):
            prev_weights = copy(self.weights)

        for i in range(n_epochs):
            prediction = predict_probabilities(self.weights, self.X)
            delta = np.matmul(
                self.X.transpose(),
                (prediction - self.y) * prediction * (1 - prediction),
            )
            self.weights -= delta * learning_rate

            if not (eps is None):
                if np.linalg.norm(self.weights - prev_weights) < eps:
                    print(f"Algorithm converged after {i + 1} iterations")
                    break
                prev_weights = copy(self.weights)

    def SGD(
        self, n_epochs=100, learning_rate=0.1, batch_size=32, random_state=None, eps=None
    ):
        # based on https://realpython.com/gradient-descent-algorithm-python/

        n_obs = self.X.shape[0]
        xy = np.c_[self.X.reshape(n_obs, -1), self.y.reshape(n_obs, 1)]

        # random number generator for permutation of observations
        rng = np.random.default_rng(seed=self.seed)

        batch_size = int(batch_size)
        if not 0 < batch_size <= n_obs:
            raise ValueError(
                "Batch size must be greater than zero and not greater than"
                "the number of observations!"
            )

        if not (eps is None):
            prev_weights = copy(self.weights)

        for i in range(n_epochs):
            rng.shuffle(xy)
            batch_division = np.arange(batch_size, n_obs, batch_size)
            xy_batches = np.split(xy, batch_division)

            for n_batch in range(len(batch_division)):
                x_batch, y_batch = (
                    xy_batches[n_batch][:, :-1],
                    xy_batches[n_batch][:, -1:],
                )
                prediction = predict_probabilities(self.weights, x_batch)
                delta = np.matmul(
                    x_batch.transpose(),
                    (prediction - y_batch) * prediction * (1 - prediction),
                )

                self.weights -= delta * learning_rate
            if not (eps is None):
                if np.linalg.norm(self.weights - prev_weights) < eps:
                    print(f"Algorithm converged after {i + 1} iterations")
                    break
                prev_weights = copy(self.weights)

    def log_likelihood(self) -> float:
        """Returns the log-likelihood value for the model, based on weights"""
        sigm = predict_probabilities(self.weights, self.X)  # sigmoid(beta * x)
        temp = self.y * np.log(sigm) + (1 - self.y) * np.log(1 - sigm)
        return float(np.sum(temp))

    def R2_measure(self) -> float:
        """Returns R2 measure defined as 1 - loglikelihood(model)/loglikelihood(null_model)"""

        log_likelihood = self.log_likelihood()
        pi = np.mean(self.y)
        null_intercept = np.log(pi/(1 - pi))
        weights_null = np.zeros(self.weights.shape)
        weights_null[0] = null_intercept
        prediction_null = predict_probabilities(weights_null, self.X)
        log_likelihood_null = np.sum(self.y * np.log(prediction_null) + (1 - self.y) * np.log(1 - prediction_null))
        return 0 if np.all(log_likelihood_null) == 0 else 1 - log_likelihood / log_likelihood_null

