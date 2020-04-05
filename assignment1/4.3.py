from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
iris = datasets.load_iris()
num_e=[5, 10, 50, 150, 200]
result=[]
X, X_test, y, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
for e in num_e:
    clf = GradientBoostingClassifier(n_estimators=e,random_state=42)
    scores=cross_val_score(clf, X, y, cv=10)
    result.append(float(scores.mean()))
best_e=num_e[result.index(max(result))]
gbc=GradientBoostingClassifier(n_estimators=best_e,random_state=42)
gbc.fit(X, y)
print(best_e)
print(accuracy_score(y_test,gbc.predict(X_test)))
plt.xlabel('number of estimators')
plt.ylabel('accuracy rate')
plt.plot(num_e, result,'black')
plt.show()


