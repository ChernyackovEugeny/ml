import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# распределение таргета + describe + skewness, kurtosis
def target_info(target, width=7, height=10):
    print(target.describe())
    print('-----------------------')
    print(f'Skewness: {target.skew():.3f}')
    print(f'Skewness: {target.kurt():.3f}')

    plt.figure(figsize=(width, height))
    sns.histplot(target, color='g', bins=100, kde=True, alpha=0.4)
    plt.show()

# numerical variables
def numfeature_target_relate(data, feature, target, width=7, height=10, ylimit=(0,800000)):
    y_col = target.name if target.name is not None else 'target'  # если у таргета нет имени
    target = target.copy()
    target.name = y_col

    data_feature = pd.concat([
        target.reset_index(drop=True),
        data[feature].reset_index(drop=True)
    ], axis=1)

    data_feature.plot.scatter(
        x=feature,
        y=y_col,
        figsize=(width, height),
        ylim=ylimit,
        title=f'{feature} vs {y_col}'
    )
    plt.show()

# categorical variables
def catfeature_target_boxplot(data, feature, target, width=16, height=8, ymin=0, ymax=800000):
    y_col = target.name if target.name is not None else 'target'
    target = target.copy()
    target.name = y_col

    data_feature = pd.concat([
        target.reset_index(drop=True),
        data[feature].reset_index(drop=True)
    ], axis=1)

    f, ax = plt.subplots(figsize=(width, height))
    fig = sns.boxplot(x=feature, y=y_col, data=data_feature)
    fig.axis(ymin=ymin, ymax=ymax)
    plt.xticks(rotation=90)
    plt.show()

# correlation matrix - heatmap style
def corrmat(data, target, width=12, height=9):
    dataset = pd.concat([
        data.reset_index(drop=True),
        target.reset_index(drop=True)
        ], axis=1)
    corrmat = dataset.corr()
    f, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(corrmat, vmax=.8, square=True, fmt='.2f')
    plt.show()

# k - number of variables for heatmap
def most_corr_heatmap(data, target, width=12, height=9, k=10):
    y_col = target.name if target.name is not None else 'target'
    target = target.copy()
    target.name = y_col

    dataset = pd.concat([
        target.reset_index(drop=True),
        data.reset_index(drop=True)
    ], axis=1)

    corrmat = dataset.corr()
    topk = corrmat[y_col].abs().sort_values(ascending=False).head(k).index
    cm = dataset[topk].corr()

    plt.figure(figsize=(width, height))
    sns.set(font_scale=1.25)
    sns.heatmap(cm, annot=True, square=True, fmt='.2f', cbar=True)
    plt.title(f'Top {k} correlated features with {y_col}')
    plt.show()






