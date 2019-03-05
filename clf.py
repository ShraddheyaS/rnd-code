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

X = np.array([(float(ele[-4]), float(ele[-3])) for ele in rows])
y = np.array([0 if ele[-1]=='L' else 1 for ele in rows])

X_train = X[0 : 4350] # 4350
X_test = X[4350 :]
y_train = y[0 : 4350]
y_test = y[4350 : ]
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X_train, y_train) for clf in models)
scores = [accuracy_score(y_test, clf.predict(X_test)) for clf in models]
print(scores)
