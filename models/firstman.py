import numpy as np
import pandas as pd

class AnaliticLinReg():
    def __init__(self):
        self.w = 0

    def fit(self, X, y):
        X = X.to_numpy()
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))
        y = y.to_numpy().reshape(-1)

        self.w = np.linalg.inv(X.T @ X) @ (X.T @ y)
        return self

    def predict(self, X):
        X = X.to_numpy()
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))

        return X @ self.w

# l1, l2 reg; stoch_des; learning_rate; cross-val; data