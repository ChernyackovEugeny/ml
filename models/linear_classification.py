# hinge, support vector machine(SVM)
import numpy as np

class LinClass():
    def __init__(self, l2, lr, s):
        self.l2 = l2
        self.lr = lr
        self.s = s
        self.weights = []
        self.loss = []

    def fit(self, X, y):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))
        self.N, self.D = X.shape

        y = np.asarray(y)

        self.w = np.random.normal(size=(self.D,))

        for i in range(self.s):
            self.w = self.w - self.lr * self.calc_grad(X, y)
            self.weights.append(self.w.copy())

            l2_coef = self.l2 * np.sum(self.w[1:] ** 2)
            los = np.sum(np.maximum(0, 1 - y * X.dot(self.w))) + l2_coef
            self.loss.append(los)

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


# многоклассовая классификация, k классов

class LogRegOVA():  # one versus all
    def __init__(self, k, l2, lr, s):
        self.k = k
        self.l2 = l2
        self.lr = lr
        self.s = s

    def fit(self, X, y):
        # метки на входе от 1 до k
        # создать k выборок, научить k штук LinClass по соответ выборкам
        # метки 1 и -1, в k-той выборке k-тый класс - единица, остальные -1

        answers = []
        for i in range(1, self.k+1):
            y_k = np.where(y == i, 1, -1)
            answers.append(y_k)
        # k выборок готовы

        self.models = []
        for i in range(0, self.k):
            model_k = LinClass(self.l2, self.lr, self.s)
            model_k.fit(X, answers[i])
            self.models.append(model_k)
        # k обученных моделей готовы, выдают знак, хотим margin, придется вытаскивать self.w
        # отличают k-тый класс от остальных, остальное на predict

    def predict(self, X):
        X = np.asarray(X)
        ones_col = np.ones((X.shape[0], 1))
        X = np.hstack((ones_col, X))

        # хотим получать класс с самым уверенным ответом - наибольшее значение в models_pred
        models_pred = []
        for i in range(self.k):
            models_pred.append(X @ self.models[i].w)
        models_pred = np.array(models_pred)  # из одномерного списка списков делаем двумерный массив - shape: (k, n_samples)

        return np.argmax(models_pred, axis=0) + 1  # выдаст индекс, но метки от 1 до k








