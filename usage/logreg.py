import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.logit_regress import LogReg

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('../data/processed/Person/person_mark1.csv')
data['Personality'] = np.where(data['Personality']==-1, 0, 1)  # чуть подгоняем метки под модель

X = data.drop(columns=['Personality'])
y = data['Personality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, stratify=y)

logreg = LogReg(0.5, 0.01, 10000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# plt.figure(figsize=(7, 10))
# plt.plot(logreg.loss, label='изменение лосса')
# plt.xlabel('итерация')
# plt.ylabel('значение лосса')
# plt.title('loss and iterations')
# plt.legend()
# plt.grid()
# plt.show()

# weights = np.array(logreg.weights)
# plt.figure(figsize=(7, 10))
# for i in range(weights.shape[1]):
#     plt.plot(weights[:, i], label=f'weight {i}')
# plt.xlabel('Iteration')
# plt.ylabel('Weight value')
# plt.title('Change of weights')
# plt.legend()
# plt.grid(True)
# plt.show()

# ---------------------

sklogreg = LogisticRegression()
sklogreg.fit(X_train, y_train)
sky_pred = sklogreg.predict(X_test)
skaccuracy = accuracy_score(y_test, sky_pred)

print(accuracy, skaccuracy)

