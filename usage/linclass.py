import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.linear_classification import LinClass

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
# support vector classificator

data = pd.read_csv('../data/processed/Person/person_mark1.csv')

X = data.drop(columns=['Personality'])
y = data['Personality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, stratify=y)

lincl = LinClass(0.1, 0.01, 100000)
lincl.fit(X_train, y_train)
y_pred = lincl.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ------------------
skclass = LinearSVC(C=1.0, max_iter=10000)
skclass.fit(X_train, y_train)
sky_pred = skclass.predict(X_test)
skaccuracy = accuracy_score(y_test, sky_pred)

print(accuracy, skaccuracy)

