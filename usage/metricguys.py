import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

from models.metric_methods import KNN
from models.metric_methods import Kernel_KNN
from models.metric_methods import KernelReg

data = pd.read_csv('../data/processed/Iris/iris_mark1.csv')

X = data.drop(columns=['Species'])
y = data['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, train_size=0.2, stratify=y)

knn = KNN(10)
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1)

ker_knn = Kernel_KNN(10, 1, 3)
ker_knn.fit(X_train, y_train)
y_pred2 = ker_knn.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)

ker_reg = KernelReg(10, 1)
ker_reg.fit(X_train, y_train)
y_pred3 = ker_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred3)

print(accuracy1, accuracy2)
print(mse)



