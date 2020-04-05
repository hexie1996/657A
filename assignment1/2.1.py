from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets
import numpy as np
import csv
from matplotlib import pyplot as plt
iris = datasets.load_iris()
parameters=[1, 5, 10, 15, 20, 25, 30, 35]
result=[]
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
for p in parameters:
    knn = KNeighborsClassifier(n_neighbors=p)
    knn.fit(X_train,y_train)
    result.append(knn.score(X_val,y_val))
best_p=parameters[result.index(max(result))]
knn = KNeighborsClassifier(n_neighbors=best_p)
knn.fit(X_train,y_train)
print(best_p,knn.score(X_test,y_test))
plt.xlabel('parameter')
plt.ylabel('accuracy rate')
plt.plot(parameters, result,'r')
plt.show()

