import pandas as pd
df = pd.read_csv('../data/raw/Iris.csv', index_col='Id')



print(df['Species'].unique())