import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

import pandas as pd
# Loading preprocessed datasets
# nba, TARGET
# bankrupcy, FLAG
dataset_name = "bankrupcy"
target_name="FLAG"
dataset=pd.read_csv(f"data/" + dataset_name+ ".csv")
X = dataset.drop(columns=target_name)
y = dataset[[target_name]]


# X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

clf = LogisticRegression(penalty="l1", 
                         solver="saga", 
                         C=X_train.shape[0] / 1e-4 , 
                         verbose=1, 
                         tol=1e-3)
clf.fit(X_train, y_train)

print(f"Accuracy on training set: {clf.score(X_train, y_train)}")
print(f"Accuracy on test set: {clf.score(scaler.transform(X_test), y_test)}")
print(f"Norm of the coefficients: {np.sum(np.abs(clf.coef_))}")
print(f"Number of non-zero coefficients: {np.sum(np.abs(clf.coef_) > 1e-4)}/{np.size(clf.coef_)}")