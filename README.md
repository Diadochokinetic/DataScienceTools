# DataScienceTools

This repository contains some modules I consider useful in Data Science tasks. Most of them are based on sklearn and require input data to be in pandas DataFrames or Series.

## Table of contents

1. [ Preprocessing ](#preprocessing) <br>
1.1 [ NumScaler ](#numscaler)

<a name="preprocessing"></a>
## 1. Preprocessing

sometext

<a name="numscaler"></a>
### 1.1 NumScaler

NumScaler is a wrapper for scalers. Some scalers only take numerical input and can't ignore non numerical data. NumScaler identifies numerical data and passes it to a Scaler (default=sklearn.preprocessing.StandardScaler).
