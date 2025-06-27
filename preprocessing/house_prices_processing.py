import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation.eda import numfeature_target_relate
from data_preparation.eda import scatter_plot
from data_preparation.eda import corrmat
from data_preparation.eda import target_outliers
from data_preparation.eda import target_norm
from data_preparation.eda import target_info

from data_preparation.data_clean import num_cols
from data_preparation.data_clean import cat_cols
from data_preparation.data_clean import do_nan_void
from data_preparation.data_clean import nan_cols

from data_preparation.data_processing import label_encode

dataset = pd.read_csv('../data/raw/house_prices/train.csv', index_col='Id')
target = dataset['SalePrice']
data = dataset.drop(columns=['SalePrice'])

# NaNs
data, target = do_nan_void(dataset, ['Electrical'], target)
features, nans_amount = nan_cols(data)
data = data.drop(columns=features)

# outliers
# print(data.sort_values(by='GrLivArea', ascending=False)[:2])
data = data.drop([524, 1299])
target = target.drop([523, 1298])

data['HasBsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
data['TotalBsmtSF'] = data['TotalBsmtSF'].apply(lambda x: np.log(x) if x > 0 else 0)

target = np.log(target)  # сдвигаем к нормальному распределению при skewness>0 - data transformation
data['GrLivArea'] = np.log(data['GrLivArea'])

cat_col = cat_cols(data)
data = label_encode(data, cat_col)

# dataset = pd.concat([data, target])
# dataset.to_csv('../data/processed/House_Prices/prices_mark1.csv', index=False)







