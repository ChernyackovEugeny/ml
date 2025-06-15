import numpy as np
from collections import Counter

# классификация, найдем k ближайших соседей, по ним определим класс таргета
class KNN():
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        # lazy learning; обучение - запомнить выборку
        self.X_tr = np.asarray(X)
        self.y_tr = np.asarray(y)

    def predict(self, X):
        # для каждого объекта k ближ соседей
        X = np.asarray(X)
        pred = []
        for index, obj in enumerate(X):
            distances = []
            for nei_index, nei in enumerate(self.X_tr):
                distances.append((nei_index, self.calc_dist(obj, nei)))
            k_nei = sorted(distances, key=lambda x: x[1])[:self.k]
            k_nei_targets = [self.y_tr[i[0]] for i in k_nei]

            # хотим самый частый класс
            count = Counter(k_nei_targets)
            pred.append(count.most_common()[0][0])
        pred = np.array(pred)
        return pred

    def calc_dist(self, x, y):
        return np.sqrt(np.sum((x - y)**2))

# хочется взвешенный knn, веса либо зависят от положения объектов в k_nei, самые юлижние больше поощеряем
# либо зависят от самого расстояния, нужна ядерная функция(kernel function)

class Kernel_KNN():
    def __init__(self, k, h, classes):
        self.k = k
        self.h = h
        self.classes = classes

    def fit(self, X, y):
        self.X_tr = np.asarray(X)
        self.y_tr = np.asarray(y)

    def predict(self, X):
        X = np.asarray(X)
        pred = []
        for index, obj in enumerate(X):
            distances = []
            for nei_index, nei in enumerate(self.X_tr):
                distances.append((nei_index, self.calc_dist(obj, nei)))
            k_nei = sorted(distances, key=lambda x: x[1])[:self.k]
            k_nei_targets = [self.y_tr[i[0]] for i in k_nei]

            ans = []
            # итерируемся по всем возможным классам (от 1 до self.classes)
            for target in range(1, self.classes + 1):
                # индексы, при которых текущий таргет совпадает с k_nei_targets
                indexes = [i for i in range(len(k_nei_targets)) if k_nei_targets[i] == target]
                # вычисляем веса по всем возможным таргетам с помощью кернел функции
                weight = np.sum([self.kernel_func(k_nei[i][1] / self.h) for i in indexes])
                ans.append(weight)
            # отбираем вес с наибольшим значением
            pred.append(np.argmax(ans) + 1)
        pred = np.array(pred)
        return pred

    def calc_dist(self, x, y):
        return np.sqrt(np.sum((x - y)**2))

    # прямоугольное ядро, пусть гладкость не важна для простоты
    def kernel_func(self, x):
        return 0.5 if np.linalg.norm(x) <= 1 else 0


# kernel regression ф-ла Надарая-Ватсона
class KernelReg():
    def __init__(self, k, h):
        self.k = k
        self.h = h

    def fit(self, X, y):
        self.X_tr = np.asarray(X)
        self.y_tr = np.asarray(y)

    def predict(self, X):
        X = np.asarray(X)
        pred = []
        for index, obj in enumerate(X):
            distances = []
            for nei_index, nei in enumerate(self.X_tr):
                distances.append((nei_index, self.calc_dist(obj, nei)))
            k_nei = sorted(distances, key=lambda x: x[1])[:self.k]
            k_nei_targets = np.array([self.y_tr[i[0]] for i in k_nei])
            ker_dist = np.array([self.kernel_func(i[1] / self.h) for i in k_nei])

            pred.append(np.sum(ker_dist * k_nei_targets) / np.sum(ker_dist))
        return pred

    def calc_dist(self, x, y):
        return np.sqrt(np.sum((x - y)**2))

    # гауссово ядро, важно в регрессии
    def kernel_func(self, x):
        return 1/np.sqrt(2*np.pi) * np.exp(-0.5 * x**2)

