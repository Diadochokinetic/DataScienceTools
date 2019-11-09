# DataScienceTools

This repository contains some modules I consider useful in Data Science tasks. Most of them are based on sklearn and require input data to be in pandas DataFrames or Series.

## Table of contents

1. [ Preprocessing ](#preprocessing) <br>
1.1 [ NumScaler ](#numscaler) <br>
1.2 [ Imputer ](#imputer)

<a name="preprocessing"></a>
## 1. Preprocessing

These modules are meant to help with preprocessing tasks. They are built upon sklearn.base.Transformermixin and can be used in sklearn.Pipeline. Possible tasks are scaling, imputing ...


<a name="numscaler"></a>
### 1.1 NumScaler

NumScaler is a wrapper for scalers. Some scalers only take numerical input and can't ignore non numerical data. NumScaler identifies numerical data and passes it to a Scaler (default=sklearn.preprocessing.StandardScaler).


<a name="imputer"></a>
### 1.2 Imputer

Imputes missing values based on 'sklearn.base.TransformerMixin'. Numerics values can be imputed by a given metric (default=np.mean) or by a constant value when passed in manual_values. Non numeric values can be imputed by the most frequent value or by a constant value when passed in manual_values.