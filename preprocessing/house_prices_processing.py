import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/raw/house_prices/train.csv', index_col='Id')
target = data['SalePrice']
data = data.drop(columns=['SalePrice'])

# figure = plt.figure(figsize=(15, 20))
# plt.barh(data.isna().sum(axis=0).index, data.isna().sum(axis=0).values)
# plt.show()

# распределение таргета
# print(target.describe())
# plt.figure(figsize=(9, 8))
# sns.histplot(target, color='g', bins=100, kde=True, alpha=0.4)
# plt.show()

# распределения численных признаков
# print(list(set(data.dtypes.tolist())))
# data_num = data.select_dtypes(include=['float64', 'int64'])
# data_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
# plt.show()

def num_cols(data):
    data_num = data.select_dtypes(include=['number'])
    data_num_features = data_num.columns.tolist()
    return data_num_features


