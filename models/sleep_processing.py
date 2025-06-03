import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

data = pd.read_csv('../data/raw/Sleep_health_and_lifestyle_dataset.csv', index_col='Person ID')

# plt.figure(figsize=(7, 10))
# plt.barh(data.isna().sum(axis=0).index, data.isna().sum(axis=0).values, color='skyblue')
# plt.show()

# Sleep Disorder 219 NaN
data['ins'] = np.where(data['Sleep Disorder'] == 'Insomnia', 1, 0)
data['apnea'] = np.where(data['Sleep Disorder'] == 'Sleep Apnea', 1, 0)
data['no_sleep_illness'] = np.where(data['Sleep Disorder'].isna(), 1, 0)

data.drop('Sleep Disorder', axis=1, inplace=True)

data['Gender'] = np.where(data['Gender'] == 'Male', 1, 0)

# for prof in data['Occupation'].unique():
#     print(prof, np.mean(data.loc[data['Occupation']==prof]['Sleep Duration']), len(data.loc[data['Occupation']==prof]))
# удалю профессии, вернуть и проверить метрики с ними

data.drop(columns=['Occupation'], inplace=True)

r = {'Normal': 0, 'Overweight': 1, 'Obese': 2, 'Normal Weight': 0}
data['BMI Category'] = data['BMI Category'].apply(lambda x: r[x])

data.drop(columns=['Blood Pressure'], inplace=True)

data.to_csv('../data/processed/Sleep/sleep_mark1.csv', index=False)

