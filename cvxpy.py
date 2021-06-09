import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

m = X_train.shape[0]
n = X_train.shape[1]

beta = cp.Variable(n)
lambd = cp.Parameter(nonneg=True)
log_likelihood = cp.sum(
    cp.multiply(y_train, X_train @ beta) - cp.logistic(X_train @ beta)
)
problem = cp.Problem(cp.Maximize(log_likelihood/m - lambd * cp.norm(beta, 1)))

def error(scores, labels):
  scores[scores > 0] = 1
  scores[scores <= 0] = 0
  return np.sum(np.abs(scores - labels)) / float(np.size(labels))

trials = 100
train_error = np.zeros(trials)
test_error = np.zeros(trials)
lambda_vals = np.logspace(-2, 0, trials)
beta_vals = []
for i in range(trials):
    lambd.value = lambda_vals[i]
    problem.solve()
    train_error[i] = error( (X_train @ beta).value, y_train)
    test_error[i] = error( (X_test @ beta).value, y_test)
    beta_vals.append(beta.value)

# plt.plot(lambda_vals, train_error, label="Train error")
# plt.plot(lambda_vals, test_error, label="Test error")
# plt.xscale("log")
# plt.legend(loc="upper left")
# plt.xlabel(r"$\lambda$", fontsize=16)
# plt.savefig("errors.png")


for i in range(n):
    plt.plot(lambda_vals, [wi for wi in beta_vals])
plt.xlabel(r"$\lambda$", fontsize=16)
plt.xscale("log")
plt.savefig("variables.png")