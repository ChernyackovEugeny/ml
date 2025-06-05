import pandas as pd
data = pd.read_csv('../data/raw/Iris.csv', index_col='Id')

print(data['Species'].unique())
