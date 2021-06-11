import numpy as np
import sys
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import SGDClassifier

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

clf = SGDClassifier(loss="log", penalty="l1", l1_ratio=1.0, verbose=1, alpha=1e-4)
clf.fit(X_train, y_train)

sys.stdout = old_stdout
loss_history = mystdout.getvalue()
loss_list = []
bias_list = []
norm_list = []
for line in loss_history.split('\n'):
    if(len(line.split("Bias: ")) == 1):
        continue
    bias_list.append(float(line.split("Bias: ")[-1].split(",")[0]))
    loss_list.append(float(line.split("loss: ")[-1]))
    norm_list.append(float(line.split("Norm: ")[-1].split(",")[0]))

plt.figure(figsize=(12,8))
plt.plot(np.arange(len(loss_list)), loss_list)
plt.xlabel("Time in epochs"); plt.ylabel("Loss")
plt.savefig("SGD_loss.png")

plt.figure(figsize=(12,8))
plt.plot(np.arange(len(bias_list)), bias_list)
plt.xlabel("Time in epochs"); plt.ylabel("Bias")
plt.savefig("SGD_bias.png")

plt.figure(figsize=(12,8))
plt.plot(np.arange(len(bias_list)), norm_list)
plt.xlabel("Time in epochs"); plt.ylabel("Norm")
plt.savefig("SGD_norm.png")

print(f"Accuracy on training set: {clf.score(X_train, y_train)}")
print(f"Accuracy on test set: {clf.score(scaler.transform(X_test), y_test)}")
print(f"Norm of the coefficients: {np.linalg.norm(clf.coef_)}")
print(f"Number of non-zero coefficients: {np.sum(clf.coef_ != 0)}/{np.size(clf.coef_)}")