import csv
import jenkspy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')

fields = []
rows = []

with open('data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)


X = np.array([(float(ele[-4])) for ele in rows])
y = np.array([0 if ele[-1]=='L' else 1 for ele in rows])

# X_train = X[0 : 4350] -- unsupervised method, shouldn't be a problem

breaks = jenkspy.jenks_breaks(X, nb_class=2)
print("Breaks: ")
print(breaks)

# accuracy
predictions = [0 if (x >= breaks[0] and x <= breaks[1]) else 1 if (x >= breaks[1] and x <= breaks[2]) else "gadbad" for x in X]
correct = 0
for i, pred in enumerate(predictions):
    if pred == y[i]:
        correct += 1
    if pred == "gadbad":
        assert(False)

print("Accuracy = " + str(float(correct / len(y))))

plt.figure(figsize = (10,8))
hist = plt.hist(X, align='left', color='g')
for b in breaks:
    plt.vlines(b, ymin=0, ymax = max(hist[0]))

plt.savefig("jenks.png")
plt.show()  
