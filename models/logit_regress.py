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

class LogRegOVA():
    def __init__(self, k, t, lr, s):
        self.k = k  # кол-во классов
        self.t = t  # порог для вероятностей
        self.lr = lr
        self.s = s

    def fit(self, X, y):
        # хотим k моделей, которые отличают i-тый класс от всех остальных
        # создаем соответствующие выборки, сразу обучаем на них модели, классы от 1 до k

        self.models = []
        for i in range(1, self.k+1):
            y_k = np.where(y == i, 1, 0)
            model_k = LogReg(self.t, self.lr, self.s)
            model_k.fit(X, y_k)
            self.models.append(model_k)

        return self

    def predict(self, X):
        # для каждого объекта хотим вектор вероятностей, находится в model.probabilities
        probs = []
        for i in range(self.k):
            pred_k = self.models[i].predict(X)
            prob_k = self.models[i].probability  # с каждой модели по вектору вероятностей принадлежности объекта к конкретному классу
            probs.append(prob_k)

        probs = np.array(probs).T  # (k_models, n_samples)
        # в i-той строке вероятности для i-того объекта

        self.probabilities = probs

        exp_probs = np.exp(probs)
        softmax_probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)
        # keepdims=True -> размерность (n, 1), а не (n, 0) для broadcasting при делении

        pred = np.argmax(softmax_probs, axis=1) + 1
        return pred