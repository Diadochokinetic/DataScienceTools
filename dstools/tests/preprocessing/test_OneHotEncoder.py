import pandas as pd
import numpy as np
import unittest
from dstools.preprocessing.OneHotEncoder import OneHotEncoder

class TestOneHotEncoder(unittest.TestCase):

    def compare_DataFrame(self, df_transformed, df_transformed_correct):

        """
        helper function to compare the values of the transformed DataFrame with the values of a correctly transformed DataFrame
        """
        #same number of columns
        self.assertEqual(len(df_transformed.columns), len(df_transformed_correct.columns))

        #check for every column in correct DataFrame, that all items are equal
        for column in df_transformed_correct.columns:

            #compare every element
            for x, y in zip(df_transformed[column], df_transformed_correct[column]):

                #if both values are np.NaN, the assertion fails, although they are equal
                if np.isnan(x)==True and np.isnan(y)==True:

                    pass
                    
                else:
                    
                    self.assertEqual(x, y)


    def test_only_non_numeric(self):

        """
        only columns containing non numerical values should be encoded
        """
        df = pd.DataFrame({'x1':[1,2], 'x2':['a','b']})
        df_transformed_correct = pd.DataFrame({'x1':[1,2], 'x2_OHE_a':[1,0], 'x2_OHE_b':[0,1]})
        df_transformed = OneHotEncoder().fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_REST_class(self):

        """
        output DataFrame should have one encoded column and one encoded REST column
        """
        df = pd.DataFrame({'x2':['a','a','b']})
        df_transformed_correct = pd.DataFrame({'x2_OHE_a':[1,1,0], 'x2_OHE_REST':[0,0,1]})
        df_transformed = OneHotEncoder(number_of_top_values=1).fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_no_REST_class(self):

        """
        output DataFrame should not contain a REST class
        """
        df = pd.DataFrame({'x2':['a','a','b']})
        df_transformed_correct = pd.DataFrame({'x2_OHE_a':[1,1,0], 'x2_OHE_b':[0,0,1]})
        df_transformed = OneHotEncoder(number_of_top_values=2).fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_only_one_value(self):

        """
        output DataFrame should contain one column with ones
        """
        df = pd.DataFrame({'x2':['a','a','a']})
        df_transformed_correct = pd.DataFrame({'x2_OHE_a':[1,1,1]})
        df_transformed = OneHotEncoder(number_of_top_values=2).fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_ignore_missing(self):

        """
        missing value should be put in REST class
        """
        df = pd.DataFrame({'x2':['a','a',np.NaN]})
        df_transformed_correct = pd.DataFrame({'x2_OHE_a':[1,1,0],'x2_OHE_REST':[0,0,1]})
        df_transformed = OneHotEncoder(number_of_top_values=2, dropna=True).fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)

    
    def test_encode_missing(self):

        """
        missing value should be encoded as own column
        """
        df = pd.DataFrame({'x2':['a','a',np.NaN]})
        df_transformed_correct = pd.DataFrame({'x2_OHE_a':[1,1,0],'x2_OHE_nan':[0,0,1]})
        df_transformed = OneHotEncoder(number_of_top_values=2, dropna=False).fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_encode_missing_as_top_value(self):

        """
        missing value should be encoded as own column
        """
        df = pd.DataFrame({'x2':['a',np.NaN,np.NaN]})
        df_transformed_correct = pd.DataFrame({'x2_OHE_nan':[0,1,1],'x2_OHE_REST':[1,0,0]})
        df_transformed = OneHotEncoder(number_of_top_values=1, dropna=False).fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)



    """
    a column that cointains only missing values is totally unnecessary
    """
    #def test_only_ignored_missing_values(self):

        #df = pd.DataFrame({'x2':[np.NaN,np.NaN,np.NaN]})
        #df_transformed_correct = pd.DataFrame({'x2':[np.NaN,np.NaN,np.NaN]})
        #df_transformed = OneHotEncoder(number_of_top_values=1, dropna=False).fit_transform(df)
        #self.compare_DataFrame(df_transformed, df_transformed_correct)

    
    #def test_only_used_missing_values(self):

        #df = pd.DataFrame({'x2':[np.NaN,np.NaN,np.NaN]})
        #df_transformed_correct = pd.DataFrame({'x2_OHE_REST':[1,1,1]})
        #df_transformed = OneHotEncoder(number_of_top_values=1, dropna=True).fit_transform(df)
        #self.compare_DataFrame(df_transformed, df_transformed_correct)

    
    def test_special_columns_less_values(self):

        """
        less columns than top values should be added
        """
        df = pd.DataFrame({'x2':['a','a','b']})
        df_transformed_correct = pd.DataFrame({'x2_OHE_a':[1,1,0], 'x2_OHE_REST':[0,0,1]})
        df_transformed = OneHotEncoder(number_of_top_values=2,special_columns={'x2':1}).fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_special_columns_more_values(self):

        """
        more columns than top values should be added
        """
        df = pd.DataFrame({'x2':['a','a','b']})
        df_transformed_correct = pd.DataFrame({'x2_OHE_a':[1,1,0], 'x2_OHE_b':[0,0,1]})
        df_transformed = OneHotEncoder(number_of_top_values=1, special_columns={'x2':2}).fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_binary_encoded_column(self):

        """
        result should be one binary encoded column
        """
        df = pd.DataFrame({'x2':['a','a','b']})
        df_transformed_correct = pd.DataFrame({'x2_a/b':[1,1,0]})
        df_transformed = OneHotEncoder(number_of_top_values=2, compress_binary=True).fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_binary_encoded_column_3_values(self):

        """
        result should be one binary encoded column
        """
        df = pd.DataFrame({'x2':['a','a','b',np.NaN]})
        df_transformed_correct = pd.DataFrame({'x2_OHE_a':[1,1,0,0], 'x2_OHE_b':[0,0,1,0],'x2_OHE_REST':[0,0,0,1]})
        df_transformed = OneHotEncoder(number_of_top_values=2, compress_binary=True, dropna=True).fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


if __name__ == '__main__':
    unittest.main()