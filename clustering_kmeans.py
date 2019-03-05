import csv
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


fields = []
rows = []

with open('data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)


X = np.array([(float(ele[-4]), float(ele[-3])) for ele in rows])
y = np.array([0 if ele[-1]=='L' else 1 for ele in rows])

kmeans = KMeans(n_clusters=2)
kmeans = kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

label_right = centroids[0][0] > centroids[1][0]
label_left = centroids[0][0] < centroids[1][0]

correct = 0
for i, trlbl in enumerate(y):
    if (labels[i] == label_left and trlbl == 0) or (labels[i] == label_right and trlbl == 1):
        correct += 1

print("accuracy = " + str(float(correct / len(y))))

plt.scatter(*zip(*X), c=labels)
plt.show()
