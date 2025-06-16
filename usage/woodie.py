from models.wooden_guys import DecisionTree
from models.wooden_guys import RandomForest

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('../data/processed/Iris/iris_mark1.csv')

X = data.drop(columns=['Species'])
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, stratify=y)

tree = DecisionTree(3, 5)
tree.fit(X_train, y_train)
y_pred1 = tree.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1)

forest = RandomForest(3)
forest.fit(X_train, y_train)
y_pred2 = forest.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)

# dot = tree.export_graphviz()
# dot.render("tree", format="png", cleanup=True)  # создаст файл tree.png
# dot.view()  # откроет в окне просмотра, если поддерживается

# for i in range(forest.trees):
#     dot = forest.export_tree(tree_index=i)
#     dot.render(f"tree{i}", format="png", cleanup=False)  # сохранит как tree0.png
#     dot.view()

# dot = forest.export_tree(tree_index=0)
# dot.render("tree0", format="png", cleanup=False)
# dot.view()

print(accuracy1, accuracy2)

