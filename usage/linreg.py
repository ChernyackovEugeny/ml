import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.firstman import AnaliticLinReg
from models.firstman import GradLinReg
from models.firstman import StoGradLinReg
from models.firstman import L1GradLinReg
from models.firstman import L2GradLinReg

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

reg = AnaliticLinReg()
data = pd.read_csv('../data/processed/Sleep/sleep_mark1.csv')
y = data['Sleep Duration']
X = data.drop(columns=['Sleep Duration'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

# ------------------

reg.fit(X_train, y_train)
y_pred1 = reg.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred1)

# -------------------

gradreg = GradLinReg(0.1, 100)
gradreg.fit(X_train, y_train)
y_pred2 = gradreg.predict(X_test)
mse2 = mean_squared_error(y_test, y_pred2)

# изменение лосса
# plt.plot(gradreg.loss)
# plt.xlabel('Итерация')
# plt.ylabel('MSE (лосс)')
# plt.title('Изменение ошибки во время обучения')
# plt.grid()
# plt.show()

# изменение весов
# weights = np.array(gradreg.weights)
# plt.figure(figsize=(10, 6))
# for i in range(weights.shape[1]):
#     plt.plot(weights[:, i], label=f'Weight {i}')
# plt.xlabel('Iteration')
# plt.ylabel('Weight value')
# plt.title('Change of weights during gradient descent')
# plt.legend()
# plt.grid(True)
# plt.show()

# -------------------

stogradreg = StoGradLinReg(100, 50, 0.1)
stogradreg.fit(X_train, y_train)
y_pred4 = stogradreg.predict(X_test)
mse4 = mean_squared_error(y_test, y_pred4)

# plt.figure(figsize=(7, 10))
# plt.plot(stogradreg.batch_loss)
# plt.xlabel('Итерация')
# plt.ylabel('MSE (лосс)')
# plt.title('Изменение ошибки во время обучения')
# plt.grid()
# plt.show()

# plt.figure(figsize=(7, 10))
# weights = np.array(stogradreg.weights)
# for i in range(stogradreg.D):
#     plt.plot(weights[:, i], label=f'weight {i}')
# plt.xlabel('Iteration')
# plt.ylabel('Weight value')
# plt.title('Change of weights during gradient descent')
# plt.legend()
# plt.grid(True)
# plt.show()

# -------------------

l1gradreg = L1GradLinReg(0.1, 0.1, 100)
l1gradreg.fit(X_train, y_train)
y_pred5 = l1gradreg.predict(X_test)
mse5 = mean_squared_error(y_test, y_pred5)

# -------------------

l2gradreg = L2GradLinReg(0.1, 0.1, 100)
l2gradreg.fit(X_train, y_train)
y_pred6 = l2gradreg.predict(X_test)
mse6 = mean_squared_error(y_test, y_pred6)

# -------------------
skreg = LinearRegression()
skreg.fit(X_train, y_train)
y_pred3 = skreg.predict(X_test)
mse3 = mean_squared_error(y_test, y_pred3)

print(round(mse1, 2), round(mse2, 2), round(mse3, 2), round(mse4, 2), round(mse5, 2), round(mse6, 2))
