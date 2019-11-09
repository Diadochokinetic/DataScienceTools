import pandas as pd
import numpy as np
import unittest
from dstools.preprocessing.Imputer import Imputer

class TestImputer(unittest.TestCase):

    def test_impute_numeric_by_mean(self):

        """
        the missing value should be imputed by the mean of the other values
        """
        df = pd.DataFrame({'numeric':[1, 2, 3, np.NaN]})
        df_transformed_correct = pd.DataFrame({'numeric':[1, 2, 3, 2]})
        im = Imputer(missing_value=np.NaN, metric_numeric=np.mean,  manual_values={})
        df_transformed = im.fit_transform(df)
        for x, y in zip(df_transformed['numeric'],df_transformed_correct['numeric']):
            self.assertEqual(x, y)

    
    def test_impute_non_numeric_by_most_frequent(self):
        
        """
        the missing value should be imputed by the most frequent value of the other values
        """
        df = pd.DataFrame({'string':['1', '2', '1', np.NaN]})
        df_transformed_correct = pd.DataFrame({'string':['1', '2', '1', '1']})
        im = Imputer(missing_value=np.NaN, manual_values={})
        df_transformed = im.fit_transform(df)
        for x, y in zip(df_transformed['string'],df_transformed_correct['string']):
            self.assertEqual(x, y)

    
    def test_impute_manual_values(self):

        """
        Manual values have the highest priority and should be the imputed values.
        """
        df = pd.DataFrame({'string':['1', '2', '1', np.NaN], 'numeric':[1, 2, 3, np.NaN]})
        df_transformed_correct = pd.DataFrame({'string':['1', '2', '1', 'beer'], 'numeric':[1, 2, 3, 42]})
        im2 = Imputer(manual_values={'string':'beer', 'numeric':42})
        df_transformed = im2.fit_transform(df)
        for x, y in zip(df_transformed['string'],df_transformed_correct['string']):
            self.assertEqual(x, y)
        for x, y in zip(df_transformed['numeric'],df_transformed_correct['numeric']):
            self.assertEqual(x, y)


if __name__ == '__main__':
    unittest.main()