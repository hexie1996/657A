from sklearn import datasets
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.svm import SVC
import numpy as np
from matplotlib import pyplot as plt
iris = datasets.load_iris()
c=[0.1, 0.5, 1, 2, 5, 10, 20, 50]
result=[]
X_val, X_test, y_val, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
for c_value in c:
    clf = SVC(C=c_value,kernel='linear', random_state=42)
    scores = cross_val_score(clf, X_val, y_val, cv=10)
    result.append(scores.mean())
best_c=c[result.index(max(result))]
if best_c==0:
    best_c=None
clf=SVC(C=best_c, random_state=42)
clf.fit(X_val,y_val)
print(best_c,clf.score(X_test,y_test))
plt.xlabel('C')
plt.ylabel('accuracy rate')
plt.plot(c, result,'b')
plt.show()