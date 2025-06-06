# hinge, support vector machine(SVM)
import numpy as np
from collections import Counter

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

class LinClassOVA():  # one versus all
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

class LinClassAVA():  # all versus all
    def __init__(self, k, l2, lr, s):
        self.k = k
        self.l2 = l2
        self.lr = lr
        self.s = s

    def fit(self, X, y):
        samples = [[0 for j in range(self.k)] for i in range(self.k)]
        for i in range(1, self.k+1):
            for j in range(i+1, self.k+1):
                mask = (y == i) | (y == j) #  | - побитовый or
                X_ij = X[mask]
                y_ij = y[mask]
                y_ij = np.where(y_ij == i, 1, -1)
                samples[i-1][j-1] = (X_ij, y_ij)


        self.models = [[0 for j in range(self.k)] for i in range(self.k)]
        for i in range(1, self.k+1):
            for j in range(i+1, self.k+1):
                model_ij = LinClass(self.l2, self.lr, self.s)
                model_ij.fit(samples[i-1][j-1][0], samples[i-1][j-1][1])
                self.models[i-1][j-1] = model_ij

        # двумерный список, С из 2 по k моделей, остальные нули

    def predict(self, X):
        voting = []
        for i in range(1, self.k+1):
            for j in range(i+1, self.k+1):
                ans = self.models[i-1][j-1].predict(X)
                ans = np.where(ans == 1, i, j)  # обновляем метки классов на настоящие
                voting.append(ans)

        # получили С из 2 по k ответов моделей, посчитаем, выберем класс с наибольшим кол-вом голосов
        voting = np.array(voting)  # (n_models, n_samples)
        voting = voting.T  # (n_samples, n_models)

        final_preds = []
        for row in voting:
            pred = Counter(row).most_common(1)[0][0]
            # Counter(row).most_common(1) вренет список [(значение, сколько раз встретилось)] самого частого класса
            final_preds.append(pred)

        return np.array(final_preds)












