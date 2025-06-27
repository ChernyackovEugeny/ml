# https://www.kaggle.com/code/juliencs/a-study-on-regression-applied-to-the-ames-dataset

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
from data_preparation.eda import most_corr_heatmap

from data_preparation.data_clean import num_cols
from data_preparation.data_clean import cat_cols
from data_preparation.data_clean import do_nan_void
from data_preparation.data_clean import nan_cols

from data_preparation.data_processing import label_encode

dataset = pd.read_csv('../data/raw/house_prices/train.csv', index_col='Id')
data = dataset.drop(columns=['SalePrice'])

# outliers
# numfeature_target_relate(data, "GrLivArea", target)
dataset = dataset[dataset['GrLivArea'] < 4000]  # 4 выброса

# requirement of the competition
target = np.log(dataset['SalePrice'])
dataset['SalePrice'] = np.log(dataset['SalePrice'])

# Handle missing values for features where median/mean or most common value doesn't make sense

# Alley : data description says NA means "no alley access"
dataset.loc[:, "Alley"] = dataset.loc[:, "Alley"].fillna("None")
# BedroomAbvGr : NA most likely means 0
dataset.loc[:, "BedroomAbvGr"] = dataset.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
dataset.loc[:, "BsmtQual"] = dataset.loc[:, "BsmtQual"].fillna("No")
dataset.loc[:, "BsmtCond"] = dataset.loc[:, "BsmtCond"].fillna("No")
dataset.loc[:, "BsmtExposure"] = dataset.loc[:, "BsmtExposure"].fillna("No")
dataset.loc[:, "BsmtFinType1"] = dataset.loc[:, "BsmtFinType1"].fillna("No")
dataset.loc[:, "BsmtFinType2"] = dataset.loc[:, "BsmtFinType2"].fillna("No")
dataset.loc[:, "BsmtFullBath"] = dataset.loc[:, "BsmtFullBath"].fillna(0)
dataset.loc[:, "BsmtHalfBath"] = dataset.loc[:, "BsmtHalfBath"].fillna(0)
dataset.loc[:, "BsmtUnfSF"] = dataset.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA most likely means No
dataset.loc[:, "CentralAir"] = dataset.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
dataset.loc[:, "Condition1"] = dataset.loc[:, "Condition1"].fillna("Norm")
dataset.loc[:, "Condition2"] = dataset.loc[:, "Condition2"].fillna("Norm")
# EnclosedPorch : NA most likely means no enclosed porch
dataset.loc[:, "EnclosedPorch"] = dataset.loc[:, "EnclosedPorch"].fillna(0)
# External stuff : NA most likely means average
dataset.loc[:, "ExterCond"] = dataset.loc[:, "ExterCond"].fillna("TA")
dataset.loc[:, "ExterQual"] = dataset.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
dataset.loc[:, "Fence"] = dataset.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
dataset.loc[:, "FireplaceQu"] = dataset.loc[:, "FireplaceQu"].fillna("No")
dataset.loc[:, "Fireplaces"] = dataset.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
dataset.loc[:, "Functional"] = dataset.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
dataset.loc[:, "GarageType"] = dataset.loc[:, "GarageType"].fillna("No")
dataset.loc[:, "GarageFinish"] = dataset.loc[:, "GarageFinish"].fillna("No")
dataset.loc[:, "GarageQual"] = dataset.loc[:, "GarageQual"].fillna("No")
dataset.loc[:, "GarageCond"] = dataset.loc[:, "GarageCond"].fillna("No")
dataset.loc[:, "GarageArea"] = dataset.loc[:, "GarageArea"].fillna(0)
dataset.loc[:, "GarageCars"] = dataset.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
dataset.loc[:, "HalfBath"] = dataset.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA most likely means typical
dataset.loc[:, "HeatingQC"] = dataset.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA most likely means 0
dataset.loc[:, "KitchenAbvGr"] = dataset.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA most likely means typical
dataset.loc[:, "KitchenQual"] = dataset.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA most likely means no lot frontage
dataset.loc[:, "LotFrontage"] = dataset.loc[:, "LotFrontage"].fillna(0)
# LotShape : NA most likely means regular
dataset.loc[:, "LotShape"] = dataset.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA most likely means no veneer
dataset.loc[:, "MasVnrType"] = dataset.loc[:, "MasVnrType"].fillna("None")
dataset.loc[:, "MasVnrArea"] = dataset.loc[:, "MasVnrArea"].fillna(0)
# MiscFeature : data description says NA means "no misc feature"
dataset.loc[:, "MiscFeature"] = dataset.loc[:, "MiscFeature"].fillna("No")
dataset.loc[:, "MiscVal"] = dataset.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
dataset.loc[:, "OpenPorchSF"] = dataset.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA most likely means not paved
dataset.loc[:, "PavedDrive"] = dataset.loc[:, "PavedDrive"].fillna("N")
# PoolQC : data description says NA means "no pool"
dataset.loc[:, "PoolQC"] = dataset.loc[:, "PoolQC"].fillna("No")
dataset.loc[:, "PoolArea"] = dataset.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA most likely means normal sale
dataset.loc[:, "SaleCondition"] = dataset.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA most likely means no screen porch
dataset.loc[:, "ScreenPorch"] = dataset.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
dataset.loc[:, "TotRmsAbvGrd"] = dataset.loc[:, "TotRmsAbvGrd"].fillna(0)
# Utilities : NA most likely means all public utilities
dataset.loc[:, "Utilities"] = dataset.loc[:, "Utilities"].fillna("AllPub")
# WoodDeckSF : NA most likely means no wood deck
dataset.loc[:, "WoodDeckSF"] = dataset.loc[:, "WoodDeckSF"].fillna(0)

# Some numerical features are actually really categories
dataset = dataset.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45",
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75",
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120",
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })

# Encode some categorical features as ordered numbers when there is information in the order
dataset = dataset.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5,
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                        "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )

# Create new features
# 1* Simplifications of existing features
dataset["SimplOverallQual"] = dataset.OverallQual.replace({1: 1, 2: 1, 3: 1,
                                                           4: 2, 5: 2, 6: 2,
                                                           7: 3, 8: 3, 9: 3, 10: 3})
dataset["SimplOverallCond"] = dataset.OverallCond.replace({1: 1, 2: 1, 3: 1,
                                                           4: 2, 5: 2, 6: 2,
                                                           7: 3, 8: 3, 9: 3, 10: 3})
dataset["SimplPoolQC"] = dataset.PoolQC.replace({1: 1, 2: 1, 3: 2, 4: 2})
dataset["SimplGarageCond"] = dataset.GarageCond.replace({1: 1, 2: 1, 3: 1,
                                                         4: 2, 5: 2})
dataset["SimplGarageQual"] = dataset.GarageQual.replace({1: 1, 2: 1, 3: 1,
                                                         4: 2, 5: 2})
dataset["SimplFireplaceQu"] = dataset.FireplaceQu.replace({1: 1, 2: 1, 3: 1,
                                                           4: 2, 5: 2})
dataset["SimplFunctional"] = dataset.Functional.replace({1: 1, 2: 1,
                                                         3: 2, 4: 2,
                                                         5: 3, 6: 3, 7: 3,
                                                         8: 4})
dataset["SimplKitchenQual"] = dataset.KitchenQual.replace({1: 1, 2: 1, 3: 1,
                                                           4: 2, 5: 2})
dataset["SimplHeatingQC"] = dataset.HeatingQC.replace({1: 1, 2: 1, 3: 1,
                                                       4: 2, 5: 2})
dataset["SimplBsmtFinType1"] = dataset.BsmtFinType1.replace({1: 1, 2: 1, 3: 1,
                                                             4: 2, 5: 2, 6: 2})
dataset["SimplBsmtFinType2"] = dataset.BsmtFinType2.replace({1: 1, 2: 1, 3: 1,
                                                             4: 2, 5: 2, 6: 2})
dataset["SimplBsmtCond"] = dataset.BsmtCond.replace({1: 1, 2: 1, 3: 1,
                                                     4: 2, 5: 2})
dataset["SimplBsmtQual"] = dataset.BsmtQual.replace({1: 1, 2: 1, 3: 1,
                                                     4: 2, 5: 2})
dataset["SimplExterCond"] = dataset.ExterCond.replace({1: 1, 2: 1, 3: 1,
                                                       4: 2, 5: 2})
dataset["SimplExterQual"] = dataset.ExterQual.replace({1: 1, 2: 1, 3: 1,
                                                       4: 2, 5: 2})

# 2* Combinations of existing features
dataset["OverallGrade"] = dataset["OverallQual"] * dataset["OverallCond"]
dataset["GarageGrade"] = dataset["GarageQual"] * dataset["GarageCond"]
dataset["ExterGrade"] = dataset["ExterQual"] * dataset["ExterCond"]
dataset["KitchenScore"] = dataset["KitchenAbvGr"] * dataset["KitchenQual"]
dataset["FireplaceScore"] = dataset["Fireplaces"] * dataset["FireplaceQu"]
dataset["GarageScore"] = dataset["GarageArea"] * dataset["GarageQual"]
dataset["PoolScore"] = dataset["PoolArea"] * dataset["PoolQC"]
dataset["SimplOverallGrade"] = dataset["SimplOverallQual"] * dataset["SimplOverallCond"]
dataset["SimplExterGrade"] = dataset["SimplExterQual"] * dataset["SimplExterCond"]
dataset["SimplPoolScore"] = dataset["PoolArea"] * dataset["SimplPoolQC"]
dataset["SimplGarageScore"] = dataset["GarageArea"] * dataset["SimplGarageQual"]
dataset["SimplFireplaceScore"] = dataset["Fireplaces"] * dataset["SimplFireplaceQu"]
dataset["SimplKitchenScore"] = dataset["KitchenAbvGr"] * dataset["SimplKitchenQual"]
dataset["TotalBath"] = dataset["BsmtFullBath"] + (0.5 * dataset["BsmtHalfBath"]) + \
                       dataset["FullBath"] + (0.5 * dataset["HalfBath"])
dataset["AllSF"] = dataset["GrLivArea"] + dataset["TotalBsmtSF"]
dataset["AllFlrsSF"] = dataset["1stFlrSF"] + dataset["2ndFlrSF"]
dataset["AllPorchSF"] = dataset["OpenPorchSF"] + dataset["EnclosedPorch"] + \
                        dataset["3SsnPorch"] + dataset["ScreenPorch"]
dataset["HasMasVnr"] = dataset.MasVnrType.replace({"BrkCmn": 1, "BrkFace": 1, "CBlock": 1,
                                                   "Stone": 1, "None": 0})
dataset["BoughtOffPlan"] = dataset.SaleCondition.replace({"Abnorml": 0, "Alloca": 0, "AdjLand": 0,
                                                          "Family": 0, "Normal": 0, "Partial": 1})

# Create new features
# 3* Polynomials on the top 10 existing features
dataset["OverallQual-s2"] = dataset["OverallQual"] ** 2
dataset["OverallQual-s3"] = dataset["OverallQual"] ** 3
dataset["OverallQual-Sq"] = np.sqrt(dataset["OverallQual"])
dataset["AllSF-2"] = dataset["AllSF"] ** 2
dataset["AllSF-3"] = dataset["AllSF"] ** 3
dataset["AllSF-Sq"] = np.sqrt(dataset["AllSF"])
dataset["AllFlrsSF-2"] = dataset["AllFlrsSF"] ** 2
dataset["AllFlrsSF-3"] = dataset["AllFlrsSF"] ** 3
dataset["AllFlrsSF-Sq"] = np.sqrt(dataset["AllFlrsSF"])
dataset["GrLivArea-2"] = dataset["GrLivArea"] ** 2
dataset["GrLivArea-3"] = dataset["GrLivArea"] ** 3
dataset["GrLivArea-Sq"] = np.sqrt(dataset["GrLivArea"])
dataset["SimplOverallQual-s2"] = dataset["SimplOverallQual"] ** 2
dataset["SimplOverallQual-s3"] = dataset["SimplOverallQual"] ** 3
dataset["SimplOverallQual-Sq"] = np.sqrt(dataset["SimplOverallQual"])
dataset["ExterQual-2"] = dataset["ExterQual"] ** 2
dataset["ExterQual-3"] = dataset["ExterQual"] ** 3
dataset["ExterQual-Sq"] = np.sqrt(dataset["ExterQual"])
dataset["GarageCars-2"] = dataset["GarageCars"] ** 2
dataset["GarageCars-3"] = dataset["GarageCars"] ** 3
dataset["GarageCars-Sq"] = np.sqrt(dataset["GarageCars"])
dataset["TotalBath-2"] = dataset["TotalBath"] ** 2
dataset["TotalBath-3"] = dataset["TotalBath"] ** 3
dataset["TotalBath-Sq"] = np.sqrt(dataset["TotalBath"])
dataset["KitchenQual-2"] = dataset["KitchenQual"] ** 2
dataset["KitchenQual-3"] = dataset["KitchenQual"] ** 3
dataset["KitchenQual-Sq"] = np.sqrt(dataset["KitchenQual"])
dataset["GarageScore-2"] = dataset["GarageScore"] ** 2
dataset["GarageScore-3"] = dataset["GarageScore"] ** 3
dataset["GarageScore-Sq"] = np.sqrt(dataset["GarageScore"])

# Differentiate numerical features (minus the target) and categorical features
categorical_features = dataset.select_dtypes(include=["object"]).columns
numerical_features = dataset.select_dtypes(exclude=["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
dataset_num = dataset[numerical_features]
dataset_cat = dataset[categorical_features]

# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in dataset : " + str(dataset_num.isnull().values.sum()))
dataset_num = dataset_num.fillna(dataset_num.median())
print("Remaining NAs for numerical features in dataset : " + str(dataset_num.isnull().values.sum()))

# Log transform of the skewed numerical features to lessen impact of outliers
from scipy.stats import skew
skewness = dataset_num.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
print(str(skewness.shape[0]) + " skewed numerical features to log transform")
skewed_features = skewness.index
dataset_num[skewed_features] = np.log1p(dataset_num[skewed_features])

# Create dummy features for categorical values via one-hot encoding
print("NAs for categorical features in dataset : " + str(dataset_cat.isnull().values.sum()))
dataset_cat = pd.get_dummies(dataset_cat)
print("Remaining NAs for categorical features in dataset : " + str(dataset_cat.isnull().values.sum()))

# Join categorical and numerical features
dataset = pd.concat([dataset_num, dataset_cat], axis=1)
print("New number of features : " + str(dataset.shape[1]))

# Partition the dataset in train + validation sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=0.3, random_state=0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))

# Standardize numerical features
from sklearn.preprocessing import StandardScaler
stdSc = StandardScaler()
X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])
X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])


