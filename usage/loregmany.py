from models.logit_regress import LogRegOVA

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('../data/processed/Iris/iris_mark1.csv')

X = data.drop(columns=['Species'])
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y)

logregova = LogRegOVA(3, 0.5, 0.01, 10000)
logregova.fit(X_train, y_train)
y_pred = logregova.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

