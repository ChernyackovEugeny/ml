import numpy as np
from scipy.special import expit

class LogReg():
    def __init__(self, t, lr, s):
        self.t = t
        self.lr = lr
        self.s = s
        self.weights = []
        self.loss = []

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

            los = -np.sum(y*np.log(self.sigmoid(X.dot(self.w))) + (1-y)*np.log(self.sigmoid(-X.dot(self.w))))
            self.loss.append(los)

    def predict(self, X):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))

        probabilities_pred = self.sigmoid(X @ self.w)
        y_pred = np.where(probabilities_pred >= self.t, 1, 0)
        self.probability = probabilities_pred
        return y_pred

    def sigmoid(self, z):
        return expit(z)  # готовая быстрая, устойчивая реализация на С

    def calc_grad(self, X, y):
        sig = self.sigmoid(X.dot(self.w))
        grad = (1/self.N) * X.T @ (sig - y)
        return grad
