# DataScienceTools

This repository contains some modules I consider useful in Data Science tasks. Most of them are based on scikit-learn and require input data to be in pandas DataFrames or Series. <br>

The package is in an early development state (0.1.1.dev0). You can check it out with:

`$ pip install DataScienceTools`

## Table of contents

1. [ Preprocessing ](#preprocessing) <br>
1.1 [ NumScaler ](#numscaler) <br>
1.2 [ Imputer ](#imputer) <br>
1.3 [ FeatureConverter ](#featureconverter) <br>
1.4 [ OneHotEncoder ](#onehotencoder) <br>
1.5 [ Bucketizer ](#bucketizer) <br>

2. [ Classifier ](#classifier) <br>
2.1 [ BestGuessClassifier ](#bestguessclassifier) <br>
2.2 [ BestGuessRegressor ](#bestguessregressor) <br>

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
### 1.4 OneHotEncoder

This module performs binary encoding on columns containing categorical data.
It assumes that all non numeric columns contain categorical data. If categorical data is encoded in numeric columns, use dstools.preprocessing.FeatureConverter to convert these values first. The maximum number of encoded values can be given globally or fine tuned for every column. Values that exceed the maximum number of encoded values are aggregated in a REST class. Missing values can be either put in the REST class or be classified as distinct value. Categrocial columns with exactly two values (including missing values) can be encoded into one column to reduce dimensionality.

<a name="bucketizer"></a>
### 1.5 Bucketizer

The Bucketizer puts numeric features into bins. The binned feature can either replace the original feature or can be created additionally. You can bin all numeric features, pass a list of the features to be binned or pass prefix of the features to be binned.


<a name="classifier"></a>
## 2. Classifier

some text


<a name="bestguessclassifier"></a>
### 2.1 BestGuessClassifier

The BestGuessClassifier creates a constant numeric best guess for a given metric, e.g. mean absolute squared error. This Classifier is for interval scaled numeric dependent variables. Use this classifier for binnend variables, otherwise use dstools.regressors.BestGuessRegressor.


<a name="bestguessregressor"></a>
### 2.2 BestGuessRegressor

The BestGuessRegressor creates a constant best guess for a given metric, e.g. mean absolute squared error. This regressor is for interval scaled numeric dependent variables. 
