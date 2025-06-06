import pandas as pd
import numpy as np

data = pd.read_csv('../data/raw/Iris.csv', index_col='Id')

for i, species in enumerate(data['Species'].unique()):
    data['Species'] = np.where(data['Species'] == species, i+1, data['Species'])

# data['Species'] = data['Species'].astype('category').cat.codes + 1

data.to_csv('../data/processed/Iris/iris_mark1.csv', index=False)