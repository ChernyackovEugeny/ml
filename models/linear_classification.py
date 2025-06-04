# hinge, support vector machine(SVM)
import numpy as np

class LinClass():
    def __init__(self, l2, lr, s):
        self.l2 = l2
        self.lr = lr
        self.s = s

    def fit(self, X, y):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))
        self.N, self.D = X.shape

        y = np.asarray(y)

        self.w = np.random.normal(size=(self.D,))

        for i in range(self.s):
            self.w = self.w - self.lr * self.calc_grad(X, y)

        return self

    def predict(self, X):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))

        return np.sign(X @ self.w)

    def calc_grad(self, X, y):
        margins = 1 - y * X.dot(self.w)
        mask = margins > 0  # логическая маска, X[mask] - те объекты, у которых margins>0
        l2_coef = 2 * self.l2 * self.w
        l2_coef[0] = 0
        # y[mask, None] все метки по той же маске, None вторым аргументам добавляет размерность для поточечного умножения
        grad = -1 * np.sum((y[mask, None] * X[mask]), axis=0) + l2_coef
        return grad
