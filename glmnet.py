from glmnet import LogitNet

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

clf = LogitNet(tol=1e-4, verbose=1)
clf.fit(X_train, y_train)

print(f"Accuracy on training set: {clf.score(X_train, y_train)}")
print(f"Accuracy on test set: {clf.score(scaler.transform(X_test), y_test)}")
print(f"Norm of the coefficients: {np.linalg.norm(clf.coef_)}")