from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
iris = datasets.load_iris()
depth=[3, 5, 10, None]
result=[]
X, X_test, y, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
for d in depth:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores=cross_val_score(clf, X, y, cv=10)
    result.append(scores.mean())
best_depth=depth[result.index(max(result))]
if best_depth==0:
    best_depth=None
dtc = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dtc.fit(X, y)
print(best_depth)
print(accuracy_score(y_test,dtc.predict(X_test)))
x=range(len(depth))
plt.xlabel('max depth')
plt.ylabel('accuracy rate')
plt.plot(x, result,'g')
plt.xticks(x, [3, 5, 10, 'None'], rotation=45)
plt.show()


