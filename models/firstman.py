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

        self.w = np.linalg.pinv(X.T @ X) @ (X.T @ y)
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
        self.loss = []
        self.weights = []

    def fit(self, X, y):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))
        self.N, self.D = X.shape

        y = np.asarray(y).reshape(-1)

        self.w = np.random.normal(size=(self.D,))

        for i in range(self.s):
            self.w = self.w - self.lr * self.calc_grad(X, y)

            self.weights.append(self.w.copy())
            # в weights отправляется ссылка на self.w, без copy там будет просто много ссылок на один и тот же динамический объект

            y_pred = X @ self.w
            los = np.mean((y - y_pred)**2)
            self.loss.append(los)

        return self

    def predict(self, X):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))

        return X @ self.w

    def calc_grad(self, X, y):
        grad = (2 / self.N) * X.T @ (X @ self.w - y)
        return grad


class StoGradLinReg():
    def __init__(self, epochs, batch, lr):
        self.batch = batch
        self.epochs = epochs
        self.lr = lr

        self.weights = []
        self.batch_loss = []

    def fit(self, X, y):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))
        self.N, self.D = X.shape

        y = np.asarray(y)

        self.w = np.random.normal(size=(self.D,))

        for i in range(self.epochs):
            perm = np.random.permutation(self.N)
            X = X[perm]
            y = y[perm]

            for j in range(0, self.N, self.batch):
                X_batch = X[j:j+self.batch, :]
                y_batch = y[j:j+self.batch]  # y - одномерный массив после asarray

                self.w = self.w - self.lr * self.calc_grad(X_batch, y_batch)

                self.weights.append(self.w.copy())
                y_pred = X @ self.w
                los = np.mean((y - y_pred)**2)  # лосс по каждому шагу, не по эпохам
                self.batch_loss.append(los)

        return self

    def predict(self, X):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))

        return X @ self.w

    def calc_grad(self, X, y):
        grad = (2 / X.shape[0]) * X.T @ (X @ self.w - y) # вычисляем средний градиент по размеру батча
        return grad


class L1GradLinReg():
    def __init__(self, l1, lr, s):
        self.l1 = l1
        self.lr = lr
        self.s = s
        self.loss = []
        self.weights = []

    def fit(self, X, y):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))
        self.N, self.D = X.shape

        y = np.asarray(y).reshape(-1)

        self.w = np.random.normal(size=(self.D,))

        for i in range(self.s):
            self.w = self.w - self.lr * self.calc_grad(X, y)

            self.weights.append(self.w.copy())

            y_pred = X @ self.w
            los = np.mean((y - y_pred)**2) + self.l1 * np.sum(np.abs(self.w[1:]))
            self.loss.append(los)

        return self

    def predict(self, X):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))

        return X @ self.w

    def calc_grad(self, X, y):
        l1_coef = self.l1 * np.sign(self.w)
        l1_coef[0] = 0
        grad = (2 / self.N) * X.T @ (X @ self.w - y) + l1_coef
        return grad

class L2GradLinReg():
    def __init__(self, l2, lr, s):
        self.l2 = l2
        self.lr = lr
        self.s = s
        self.loss = []
        self.weights = []

    def fit(self, X, y):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))
        self.N, self.D = X.shape

        y = np.asarray(y).reshape(-1)

        self.w = np.random.normal(size=(self.D,))

        for i in range(self.s):
            self.w = self.w - self.lr * self.calc_grad(X, y)

            self.weights.append(self.w.copy())

            y_pred = X @ self.w
            los = np.mean((y - y_pred)**2) + self.l2 * np.sum(self.w[1:]**2)
            self.loss.append(los)

        return self

    def predict(self, X):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))

        return X @ self.w

    def calc_grad(self, X, y):
        l2_coef = 2 * self.l2 * self.w
        l2_coef[0] = 0
        grad = (2 / self.N) * X.T @ (X @ self.w - y) + l2_coef
        return grad

# cross-val; data
# реализация lasso, ridge в sklearn
# новые данные для классификации