import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)
data = pd.read_csv('../data/raw/personality_dataset.csv')

data.dropna(inplace=True)

data['Stage_fear'] = np.where(data['Stage_fear']=='Yes', 1, 0)
data['Drained_after_socializing'] = np.where(data['Drained_after_socializing']=='Yes', 1, 0)
data['Personality'] = np.where(data['Personality']=='Introvert', 1, -1)

data.to_csv('../data/processed/Person/person_mark1.csv', index=False)
# не добавлять индексную колонку

