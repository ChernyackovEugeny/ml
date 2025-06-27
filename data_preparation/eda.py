import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm

# распределение таргета + describe + skewness, kurtosis
def target_info(target, width=7, height=10):
    print(target.describe())
    print('-----------------------')
    print(f'Skewness: {target.skew():.3f}')
    print(f'Kurtosis: {target.kurt():.3f}')

    plt.figure(figsize=(width, height))
    sns.histplot(target, color='g', bins=100, kde=True, alpha=0.4)
    plt.show()

# numerical variables
def numfeature_target_relate(data, feature, target, width=None, height=None, ylim=None):
    y_col = target.name if target.name is not None else 'target'  # если у таргета нет имени
    target = target.copy()
    target.name = y_col

    data_feature = pd.concat([
        target.reset_index(drop=True),
        data[feature].reset_index(drop=True)
    ], axis=1)

    fig = None if (width, height) == (None, None) else (width, height)
    data_feature.plot.scatter(
        x=feature,
        y=y_col,
        figsize=fig,
        ylim=ylim,
        title=f'{feature} vs {y_col}'
    )
    plt.show()

def scatter_plot(data, feature, target):
    y_col = target.name if target.name is not None else 'target'  # если у таргета нет имени
    target = target.copy()
    target.name = y_col

    data_feature = pd.concat([
        target.reset_index(drop=True),
        data[feature].reset_index(drop=True)
    ], axis=1)

    sns.set()
    sns.pairplot(data_feature, height=2.5)
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


# find outliers
def target_outliers(target, n=10):
    scaler = StandardScaler()
    scaled_target = scaler.fit_transform(target.values.reshape(-1, 1)).flatten()

    low_idx = scaled_target.argsort()[:n]
    high_idx = scaled_target.argsort()[-n:]

    print('outer range (low) of the distribution:')
    print(scaled_target[low_idx])

    print('\nouter range (high) of the distribution:')
    print(scaled_target[high_idx])


# statistical assumptions for multivariate techniques
# normality
def target_norm(target, width=7, height=10):
    # Histogram with normal distribution fit
    sns.histplot(target, kde=True, stat="density")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, target.mean(), target.std())
    plt.plot(x, p, 'k', linewidth=2)
    plt.title('Histogram of SalePrice with Normal PDF')
    plt.show()

    # Q-Q plot (normal probability plot)
    fig = plt.figure()
    res = stats.probplot(target, dist="norm", plot=plt)
    plt.title('Normal Q-Q Plot')
    plt.show()

# homoscedasticity(гомоскедантичность)
# Departures from an equal dispersion are shown by such shapes as cones (small dispersion at one side of the graph,
# large dispersion at the opposite side) or diamonds (a large number of points at the center of the distribution).
# scatter plots











