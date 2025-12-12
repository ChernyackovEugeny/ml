import pandas as pd
import numpy as np

from models.firstman import AnaliticLinReg
from models.firstman import GradLinReg
from models.firstman import StoGradLinReg
from models.firstman import L1GradLinReg
from models.firstman import L2GradLinReg

from data_preparation.data_clean import nan_cols

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('../data/processed/House_Prices/prices_mark1.csv')

X = dataset.drop(columns=['SalePrice'])
y = dataset['SalePrice']
anlinreg = AnaliticLinReg()
anlinreg.fit(X, y)

X_test = pd.read_csv('../data/processed/House_Prices/test.csv')
y_pred = anlinreg.predict(X_test)

submission = pd.DataFrame({
    'Id': X_test.index,
    'SalePrice': y_pred
})

submission.to_csv('submission.csv', index=False, encoding='utf-8')



