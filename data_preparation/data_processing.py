import pandas as pd
import numpy as np

# label encoding
def label_encode(data, features):
    data = data.copy()
    for feature in features:
        unique = data[feature].unique()
        mapping = {val:ind for ind, val in enumerate(unique)}
        data[feature] = data[feature].map(mapping)  # map из pandas, проходит по значениям, заменяет на mapping[value],
        # если такой ключ есть, NaN в противном случае
    return data

# one hot encoding
def ohe(data, features):
    data = data.copy()
    for feature in features:
        dummies = pd.get_dummies(data[feature], prefix=feature, prefix_sep='_')
        data = pd.concat([data.drop(columns=feature), dummies], axis=1)
    return data