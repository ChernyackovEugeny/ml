import random
import numpy as np
from collections import Counter
from graphviz import Digraph

# дерево решений, критерий остановки ветвления - гиперпараметр - минимальное кол-во объектов в листе
# классы от 1 до k, указываем при инициализации
class DecisionTree():
    class Node():
        def __init__(self, j=None, t=None, left=None, right=None, targets=None):
            self.j = j
            self.t = t
            self.left = left  # левое поддерево
            self.right = right  # правое поддерево
            self.targets = targets  # таргеты объектов, сохраняем для листов для предсказаний

    def __init__(self, classes, k=5):
        self.classes = classes
        self.k = k

    def fit(self, X, y):
        # на каждом шаге хотим самое эффективное разбиение
        # разбиваем текущий лист на 2, если уменьшаем неопределенность

        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self.maketree(X, y)

    def predict(self, X):
        # в листах метки, пропустить каждый объект через дерево, выдать вектор вероятностей в листе
        X = np.asarray(X)
        pred = []
        for obj in X:
            cur_node = self.root
            while cur_node.targets is None:
                j, t = cur_node.j, cur_node.t
                if obj[j] > t:
                    cur_node = cur_node.right
                else:
                    cur_node = cur_node.left

            targets = cur_node.targets
            ans = []
            for k in range(1, self.classes + 1):
                proba = len([i for i in targets if i == k]) / len(targets)
                ans.append(proba)
            pred.append(ans)
        pred = np.asarray(pred)

        self.probabilities = pred
        pred_targets = np.argmax(pred, axis=1) + 1

        return pred_targets

    def maketree(self, X, y):
        # переберем признаки и пороги, выберем те, которые минимизируют лосс
        n, d = X.shape

        if len(set(y)) == 1:  # если все метки одинаковые, делить бессмысленно
            return self.Node(targets=y)

        if len(y) < 2 * self.k:  # если мало объектов тоже прекращаем деление
            return self.Node(targets=y)

        min_loss = float('inf')
        opt_values = (None, None)
        opt_Right = opt_Right_t = opt_Left = opt_Left_t = None

        for j in range(d):  # по признакам
            for i in range(n):  # по объектам X, для порога
                t = X[i][j]  # порог
                idx_right = [i for i in range(n) if X[i][j] > t]
                idx_left = [i for i in range(n) if X[i][j] <= t]

                if len(idx_right) < self.k or len(idx_left) < self.k:
                    continue

                Right = X[idx_right]
                Right_t = y[idx_right]
                Left = X[idx_left]
                Left_t = y[idx_left]

                loss = (len(Right)/len(X)) * self.H(Right_t) + (len(Left)/len(X)) * self.H(Left_t)

                if loss < min_loss:
                    min_loss = loss
                    opt_values = (j, t)
                    opt_Right, opt_Right_t = Right, Right_t
                    opt_Left, opt_Left_t = Left, Left_t
        # перебрали, сохранили оптимальные значения

        if opt_values[0] is None:  # не обновили оптимальное значение ни разу, не ветвимся
            return self.Node(targets=y)

        left_node = self.maketree(np.array(opt_Left), np.array(opt_Left_t))
        right_node = self.maketree(np.array(opt_Right), np.array(opt_Right_t))
        return self.Node(j=opt_values[0], t=opt_values[1], left=left_node, right=right_node)

    def H(self, y):  # информативность, мера неоднородности, пусть будет энтропия
        count = Counter(y)
        entropy = 0
        for ind, val in count.items():
            prob_k = val / len(y)
            entropy -= prob_k * np.log(prob_k + 1e-12)
        return entropy

    def export_graphviz(self):
        dot = Digraph()
        node_id = [0]  # используем список как изменяемое число

        def add_nodes_edges(node, parent_id=None, edge_label=""):
            cur_id = node_id[0]
            node_id[0] += 1

            if node.targets is not None:
                label = f"Leaf\n" + "\n".join(f"{cls}: {count}" for cls, count in Counter(node.targets).items())
                dot.node(str(cur_id), label, shape='box', style='filled', fillcolor='lightgray')
            else:
                label = f"x[{node.j}] ≤ {node.t:.2f}"
                dot.node(str(cur_id), label)

            if parent_id is not None:
                dot.edge(str(parent_id), str(cur_id), label=edge_label)

            if node.left:
                add_nodes_edges(node.left, cur_id, "≤")
            if node.right:
                add_nodes_edges(node.right, cur_id, ">")

        add_nodes_edges(self.root)
        return dot



# случайный лес для классификации (выбираем sqrt(d) признаков)
class RandomForest():
    class Node():
        def __init__(self, j=None, t=None, left=None, right=None, targets=None):
            self.j = j
            self.t = t
            self.left = left  # левое поддерево
            self.right = right  # правое поддерево
            self.targets = targets  # таргеты объектов, сохраняем для листов для предсказаний

    def __init__(self, classes, trees=10, k=5):
        self.classes = classes  # количество классов
        self.trees = trees  # количество деревьев в ансамбле
        self.k = k  # глубина для каждого дерева

    def fit(self, X, y):
        # на каждом шаге хотим самое эффективное разбиение
        # разбиваем текущий лист на 2, если уменьшаем неопределенность
        # лосс считаем по определенному количеству случайно выбранных признаках

        X = np.asarray(X)
        y = np.asarray(y)
        n = X.shape[0]

        self.roots = []  # корни деревьев ансамбля
        for i in range(self.trees):
            ind = np.random.choice(n, size=n, replace=True)  # bootstrap
            X_boot = X[ind]
            y_boot = y[ind]
            tree_k = self.maketree(X_boot, y_boot)
            self.roots.append(tree_k)

    def predict(self, X):
        # каждое дерево выдаст класс, ответ - наиболее частый класс
        X = np.asarray(X)
        pred_proba = []  # финальный массив усредненных вероятностей
        for obj in X:
            proba = []  # вероятности одного объекта от всех деревьев
            for tree in range(self.trees):
                cur_node = self.roots[tree]
                while cur_node.targets is None:
                    j, t = cur_node.j, cur_node.t
                    if obj[j] > t:
                        cur_node = cur_node.right
                    else:
                        cur_node = cur_node.left

                targets_t = cur_node.targets
                ans_t = []  # (k, 1) вектор вероятностей для каждого класса одного дерева
                for k in range(1, self.classes + 1):
                    proba_k = len([i for i in targets_t if i == k]) / len(targets_t)
                    ans_t.append(proba_k)
                proba.append(ans_t)
            proba = np.array(proba)
            pred_proba_obj = np.mean(proba, axis=0)  # усредняем вероятности от всех деревьев для одного объекта
            pred_proba.append(pred_proba_obj)

        self.probabilities = pred_proba
        pred_classes = np.argmax(pred_proba, axis=1) + 1

        return pred_classes

    def maketree(self, X, y):
        # переберем признаки и пороги, выберем те, которые минимизируют лосс
        n, d = X.shape

        # смотрим на sqrt(d) случайных признака
        features = random.sample(range(d), round(np.sqrt(d)))  # создаем индексы

        if len(set(y)) == 1:  # если все метки одинаковые, делить бессмысленно
            return self.Node(targets=y)

        if len(y) < 2 * self.k:  # если мало объектов тоже прекращаем деление
            return self.Node(targets=y)

        min_loss = float('inf')
        opt_values = (None, None)
        opt_Right = opt_Right_t = opt_Left = opt_Left_t = None

        for j in features:  # по случайным признакам
            for i in range(n):  # по объектам X, для порога
                t = X[i][j]  # порог
                idx_right = [i for i in range(n) if X[i][j] > t]
                idx_left = [i for i in range(n) if X[i][j] <= t]

                if len(idx_right) < self.k or len(idx_left) < self.k:
                    continue

                Right = X[idx_right]
                Right_t = y[idx_right]
                Left = X[idx_left]
                Left_t = y[idx_left]

                loss = (len(Right) / len(X)) * self.H(Right_t) + (len(Left) / len(X)) * self.H(Left_t)

                if loss < min_loss:
                    min_loss = loss
                    opt_values = (j, t)
                    opt_Right, opt_Right_t = Right, Right_t
                    opt_Left, opt_Left_t = Left, Left_t
        # перебрали, сохранили оптимальные значения

        if opt_values[0] is None:  # не обновили оптимальное значение ни разу, не ветвимся
            return self.Node(targets=y)

        left_node = self.maketree(np.array(opt_Left), np.array(opt_Left_t))
        right_node = self.maketree(np.array(opt_Right), np.array(opt_Right_t))
        return self.Node(j=opt_values[0], t=opt_values[1], left=left_node, right=right_node)

    def H(self, y):  # информативность, мера неоднородности, пусть будет энтропия
        count = Counter(y)
        entropy = 0
        for ind, val in count.items():
            prob_k = val / len(y)
            entropy -= prob_k * np.log(prob_k + 1e-12)
        return entropy

    # изобразить отдельные деревья
    def export_tree(self, root=None, tree_index=0):
        if root is None:
            root = self.roots[tree_index]

        dot = Digraph()
        node_id = 0

        def traverse(node):
            nonlocal node_id
            cur_id = str(node_id)
            node_id += 1

            if node.targets is not None:
                # лист
                label = f'samples = {len(node.targets)}\n'
                count = Counter(node.targets)
                for cls in range(1, self.classes + 1):
                    label += f'{cls}: {count.get(cls, 0)}\n'
                dot.node(cur_id, label, shape='box', style='filled', fillcolor='lightgrey')
                return cur_id

            label = f'x[{node.j}] <= {node.t:.2f}'
            dot.node(cur_id, label)

            left_id = traverse(node.left)
            right_id = traverse(node.right)

            dot.edge(cur_id, left_id, label='no')
            dot.edge(cur_id, right_id, label='yes')

            return cur_id

        traverse(root)
        return dot




