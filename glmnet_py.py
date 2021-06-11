import glmnet
from glmnet.logistic import LogitNet

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

import pandas as pd
# Loading preprocessed datasets
# nba, TARGET
# bankrupcy, FLAG
dataset_name = "nba"
dataset=pd.read_csv(f"data/" + dataset_name+ ".csv")
target_name="TARGET"
X = dataset.drop(columns=target_name)
y = dataset[[target_name]]


# X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

clf = glmnet.LogitNet(tol=1e-3, verbose=1, alpha=1)
clf.fit(X_train, y_train)

print(f"Accuracy on training set: {clf.score(X_train, y_train)}")
print(f"Accuracy on test set: {clf.score(scaler.transform(X_test), y_test)}")
print(f"Norm of the coefficients: {np.sum(np.abs(clf.coef_))}")
print(f"Number of non-zero coefficients: {np.sum(np.abs(clf.coef_) > 1e-4)}/{np.size(clf.coef_)}")

coeffs = clf.coef_path_
lambdas = clf.lambda_path_
lambda_best = clf.lambda_best_
norms = np.array([np.sum(np.abs(coeffs[0, :, i])) for i in range(np.size(lambdas))])

print(f"Best Lambda: {lambda_best}")

plt.figure(figsize = (16, 12))
plt.plot(norms, coeffs[0].T)
plt.xlabel("Norm of weight vector"); plt.ylabel("Coefficients")
plt.savefig("glmnet_coeffs.png")

plt.figure(figsize = (16, 12))
plt.plot(lambdas, coeffs[0].T)
plt.xlabel("Lambda"); plt.ylabel("Coefficients")
plt.savefig("glmnet_lambda.png")

plt.figure(figsize = (16, 12))
plt.plot(range(np.size(lambdas)), coeffs[0].T)
plt.xlabel("Lambda Number"); plt.ylabel("Coefficients")
plt.savefig("glmnet_lambda_nr.png")


plt.figure(figsize = (16, 12))
plt.plot(range(np.size(lambdas)), norms)
plt.xlabel("Lambda Number"); plt.ylabel("Norm")
plt.savefig("glmnet_norms.png")