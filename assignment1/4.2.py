from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import seaborn as sns;sns.set()
from sklearn.metrics import accuracy_score
import numpy as np
iris = datasets.load_iris()
depth=[3, 5, 10, None]
num_of_trees=[5, 10, 50, 150, 200]
result=[]
X, X_test, y, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
for n in num_of_trees:
    result1 = []
    for d in depth:
        clf = RandomForestClassifier(max_depth=d,n_estimators=n, random_state=42)
        scores=cross_val_score(clf, X, y, cv=10)
        result1.append(scores.mean())
    result.append(result1)
max=0
max_i=0
max_j=0
for i in range(len(result)):
    for j in range(len(result[i])):
        if result[i][j]>max:
            max=result[i][j]
            max_i=i
            max_j=j
rfc=RandomForestClassifier(max_depth=depth[max_j],n_estimators=num_of_trees[max_i], random_state=42)
rfc.fit(X,y)
print(depth[max_j],num_of_trees[max_i])
print(accuracy_score(y_test,rfc.predict(X_test)))
x_axis_labels = [3, 5, 10, 'None']
y_axis_labels=[5, 10, 50, 150, 200]
f, ax = plt.subplots(figsize=(9, 6))
ax=sns.heatmap(result, xticklabels=x_axis_labels, yticklabels=y_axis_labels,cmap="Greys")
plt.xlabel("Tree depth")
plt.ylabel("number of trees")
plt.show()