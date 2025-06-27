import pandas as pd
import numpy as np

# весь датафрейм -> список названия столбцов с NaN-ами, список кол-ва NaN-ов
def nan_cols(data):
    nan_count = data.isna().sum()
    features = nan_count[nan_count > 0].index.tolist()
    nans_amount = nan_count[nan_count > 0].values.tolist()
    return features, nans_amount

# список - названия столбцов с категориальными признаками
def cat_cols(data):
    data_num = data.select_dtypes(include=['object', 'category'])
    data_num_features = data_num.columns.tolist()
    return data_num_features

# список - названия столбцов с численными признаками
def num_cols(data):
    data_num = data.select_dtypes(include=['number'])
    data_num_features = data_num.columns.tolist()
    return data_num_features

# данные, список фич для работы -> новые обработанные данные
def do_nan_fill(data, features, method='mean'):
    data = data.copy()
    for feature in features:
        if method == 'mean':
            assert np.issubdtype(data[feature].dtype, np.number), f"{feature} — категориальный"
            value = data[feature].mean()
        elif method == 'median':
            assert np.issubdtype(data[feature].dtype, np.number), f"{feature} — категориальный"
            value = data[feature].median()
        elif method == 'mode':
            mode = data[feature].mode(dropna=True)
            if mode.empty:
                continue
            value = mode.iloc[0]
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        data[feature] = data[feature].fillna(value)
    return data


def do_nan_void(dataset, features, target):
    dataset = dataset.copy()
    target = target.copy()

    y_col = target.name if target.name is not None else 'target'
    target.name = y_col

    dataset[y_col] = target
    dataset = dataset.dropna(subset=features)

    return dataset.drop(columns=[y_col]), dataset[y_col]


