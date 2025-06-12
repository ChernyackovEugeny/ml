from models.linear_classification import LinClassOVA
from models.linear_classification import LinClassAVA

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('../data/processed/Iris/iris_mark1.csv')

X = data.drop(columns=['Species'])
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y)

linclassova = LinClassOVA(3, 0.1, 0.001, 1000)
linclassova.fit(X_train, y_train)
y_pred1 = linclassova.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1)

linclassava = LinClassAVA(3, 0.1, 0.001, 1000)
linclassava.fit(X_train, y_train)
y_pred2 = linclassava.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print(accuracy1, accuracy2)

