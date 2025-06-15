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

    # пытаемся нарисовать
    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root
        if node.targets is not None:
            print("  " * depth + f"Leaf: {Counter(node.targets)}")
        else:
            print("  " * depth + f"Split: x[{node.j}] <= {node.t:.3f}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

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






