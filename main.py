from models.firstman import AnaliticLinReg
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error

reg = AnaliticLinReg()
data = pd.read_csv('data/processed/Sleep/sleep_mark1.csv')
y = data['Sleep Duration']
X = data.drop(columns=['Sleep Duration'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}")



