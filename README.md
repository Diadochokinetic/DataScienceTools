# DataScienceTools

This repository contains some modules I consider useful in Data Science tasks. Most of them are based on sklearn and require input data to be in pandas DataFrames or Series.

## Table of contents

1. [ Preprocessing ](#preprocessing) <br>
1.1 [ NumScaler ](#numscaler) <br>
1.2 [ Imputer ](#imputer) <br>
1.3 [ FeatureConverter ](#featureconverter) <br>
1.3 [ OneHotEncoder ](#onehotencoder)

<a name="preprocessing"></a>
## 1. Preprocessing

These modules are meant to help with preprocessing tasks. They are built upon sklearn.base.Transformermixin and can be used in sklearn.Pipeline. Possible tasks are scaling, imputing ...


<a name="numscaler"></a>
### 1.1 NumScaler

NumScaler is a wrapper for scalers. Some scalers only take numerical input and can't ignore non numerical data. NumScaler identifies numerical data and passes it to a Scaler (default=sklearn.preprocessing.StandardScaler).


<a name="imputer"></a>
### 1.2 Imputer

Imputes missing values based on 'sklearn.base.TransformerMixin'. Numerics values can be imputed by a given metric (default=np.mean) or by a constant value when passed in manual_values. Non numeric values can be imputed by the most frequent value or by a constant value when passed in manual_values.


<a name="featureconverter"></a>
### 1.3 FeatureConverter

The FeatureConverter helps to integrate common preprocessing steps into sklearn pipelines. Supported preprocessing steps are replacing values, converting to str, int or float, dropping columns or adding flags. Steps will be performed in the following order: create flags, replace values, convert types, drop columns


<a name="onehotencoder"></a>
### 1.3 OneHotEncoder

This module performs binary encoding on columns containing categorical data.
It assumes that all non numeric columns contain categorical data. If categorical data is encoded in numeric columns, use dstools.preprocessing.FeatureConverter to convert these values first. The maximum number of encoded values can be given globally or fine tuned for every column. Values that exceed the maximum number of encoded values are aggregated in a REST class. Missing values can be either put in the REST class or be classified as distinct value. Categrocial columns with exactly two values (including missing values) can be encoded into one column to reduce dimensionality.