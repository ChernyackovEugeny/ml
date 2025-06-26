import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_preparation.eda import numfeature_target_relate
from data_preparation.eda import scatter_plot
from data_preparation.eda import corrmat
from data_preparation.eda import target_outliers

from data_preparation.data_clean import num_cols
from data_preparation.data_clean import do_nan_void
from data_preparation.data_clean import nan_cols

data = pd.read_csv('../data/raw/house_prices/train.csv', index_col='Id')
target = data['SalePrice']
data = data.drop(columns=['SalePrice'])

# NaNs
data = do_nan_void(data, ['Electrical'])
features, nans_amount = nan_cols(data)
data = data.drop(columns=features)

# outliers
# print(data.sort_values(by='GrLivArea', ascending=False)[:2])
data = data.drop(data[data['Id'] == 1299].index)
data = data.drop(data[data['Id'] == 524].index)


# target_outliers(target)
# cols = num_cols(data)
# corrmat(data[cols], target)

