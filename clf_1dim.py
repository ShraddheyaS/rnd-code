import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import csv

fields = []
rows = []

with open('data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

X = np.array([(float(ele[-4])) for ele in rows])
y = np.array([0 if ele[-1]=='L' else 1 for ele in rows])

X_train = X[0 : 4350] # 4350
X_train = X_train.reshape(-1, 1)
X_test = X[4350 :]
X_test = X_test.reshape(-1, 1)
y_train = y[0 : 4350]
y_test = y[4350 : ]

C = 1.0  # SVM regularization parameter
clf = svm.LinearSVC(C=C)
clf.fit(X_train, y_train)

print("Accuracy on testing data ---")
print(accuracy_score(y_test, clf.predict(X_test)))

print("Accuracy on self ---")
X_test = X[0 : 4350]
X_test = X_test.reshape(-1, 1)
y_test = y[0 : 4350]
print(accuracy_score(y_test, clf.predict(X_test)))


print("Coefficient: ")
print(clf.coef_)
print("Intercept: ")
print(clf.intercept_)
