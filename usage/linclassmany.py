from models.linear_classification import LogRegOVA

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('../data/processed/Iris/iris_mark1.csv')

X = data.drop(columns=['Species'])
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y)

logregova = LogRegOVA(3, 0.1, 0.001, 1000)
logregova.fit(X_train, y_train)
y_pred = logregova.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

