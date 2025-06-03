from models.firstman import AnaliticLinReg
from models.firstman import GradLinReg
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

reg = AnaliticLinReg()
data = pd.read_csv('data/processed/Sleep/sleep_mark1.csv')
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

print(round(mse1, 2), round(mse2, 2))



