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

# 0.01, 0.001, 100
lincl = LinClass(0.001, 0.000001, 10000)
lincl.fit(X_train, y_train)
y_pred = lincl.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# plt.figure(figsize=(7, 10))
# plt.plot(lincl.loss, label='изменение лосса')
# plt.xlabel('итерация')
# plt.ylabel('значение лосса')
# plt.title('loss and iterations')
# plt.legend()
# plt.grid()
# plt.show()
#
# weights = np.array(lincl.weights)
# plt.figure(figsize=(7, 10))
# for i in range(weights.shape[1]):
#     plt.plot(weights[:, i], label=f'weight {i}')
# plt.xlabel('Iteration')
# plt.ylabel('Weight value')
# plt.title('Change of weights')
# plt.legend()
# plt.grid(True)
# plt.show()

# ------------------

skclass = LinearSVC(C=1.0, max_iter=10000)
skclass.fit(X_train, y_train)
sky_pred = skclass.predict(X_test)
skaccuracy = accuracy_score(y_test, sky_pred)

print(accuracy, skaccuracy)

