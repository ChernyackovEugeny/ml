import numpy as np
import pandas as pd

# метод наименьших квадратов точное решение
class AnaliticLinReg():
    def __init__(self):
        pass

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

# мнк приближенные методы(град спуски)
class GradLinReg():
    def __init__(self, lr, s):
        self.lr = lr
        self.s = s

    def fit(self, X, y):
        X = X.to_numpy()
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))
        y = y.to_numpy().reshape(-1)

        self.D = X.shape[1]
        self.N = X.shape[0]
        self.w = np.random.normal(size=(self.D,))

        for i in range(self.s):
            self.w = self.w - self.lr * self.calc_grad(X, y)

        return self

    def predict(self, X):
        X = X.to_numpy()
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))

        return X @ self.w

    def calc_grad(self, X, y):
        grad = (2 / self.N) * X.T @ (X @ self.w - y)
        return grad


# l1, l2 reg; stoch_des; learning_rate; cross-val; data