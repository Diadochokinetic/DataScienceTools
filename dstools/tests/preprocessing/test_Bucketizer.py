import pandas as pd
import numpy as np
import unittest
from dstools.preprocessing.Bucketizer import Bucketizer

class TestBucketizer(unittest.TestCase):

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
                #if np.isnan(x)==True and np.isnan(y)==True: --> doesn't work with strings
                if pd.isnull(x)==True and pd.isnull(y)==True:
                    pass
                    
                else:
                    
                    self.assertEqual(x, y)


    def test_no_numeric_feature(self):

        """
        no transformation should be performed
        """
        df=pd.DataFrame({'x':[np.NaN,'b','c']})
        df_transformed_correct=pd.DataFrame({'x':[np.NaN,'b','c']})
        bucket = Bucketizer(features=['x'])
        df_transformed = bucket.fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_one_numeric_feature_no_transformation(self):

        """
        no transformation should be performed
        """
        df=pd.DataFrame({'x':[1,2,3]})
        df_transformed_correct=pd.DataFrame({'x':[1,2,3]})
        bucket = Bucketizer()
        df_transformed = bucket.fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_one_numeric_feature_with_transformation(self):

        """
        transformation should be performed
        """
        df=pd.DataFrame({'x':[1,2,3]})
        df_transformed_correct=pd.DataFrame({'x':[0,0,0]})
        bucket = Bucketizer(bins=1, features=['x'])
        df_transformed = bucket.fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_one_numeric_feature_all_numeric(self):

        """
        transformation should be performed on all numeric columns
        """
        df=pd.DataFrame({'x':[1,2,3]})
        df_transformed_correct=pd.DataFrame({'x':[0,0,0]})
        bucket = Bucketizer(bins=1, bin_numeric=True)
        df_transformed = bucket.fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_one_numeric_feature_prefix(self):

        """
        transformation should be performed on y
        """
        df=pd.DataFrame({'x':[1,2,3], 'y':[1,2,3]})
        df_transformed_correct=pd.DataFrame({'x':[0,0,0], 'y':[1,2,3]})
        bucket = Bucketizer(bins=1, bin_numeric=False, prefix='x')
        df_transformed = bucket.fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_two_numeric_one_feature_passed(self):

        """
        transformation should be performed on y
        """
        df=pd.DataFrame({'x':[1,2,3], 'y':[1,2,3]})
        df_transformed_correct=pd.DataFrame({'x':[0,0,0], 'y':[1,2,3]})
        bucket = Bucketizer(bins=1, features=['x'])
        df_transformed = bucket.fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_two_numeric_all_numeric(self):

        """
        transformation should be performed on y
        """
        df=pd.DataFrame({'x':[1,2,3], 'y':[1,2,3]})
        df_transformed_correct=pd.DataFrame({'x':[0,0,0], 'y':[0,0,0]})
        bucket = Bucketizer(bins=1, bin_numeric=True)
        df_transformed = bucket.fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_two_numeric_one_prefix(self):

        """
        transformation should be performed on y
        """
        df=pd.DataFrame({'x':[1,2,3], 'y':[1,2,3]})
        df_transformed_correct=pd.DataFrame({'x':[1,2,3], 'y':[0,0,0]})
        bucket = Bucketizer(bins=1, prefix='y')
        df_transformed = bucket.fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_two_bins(self):

        """
        transformation should be performed on y
        """
        df=pd.DataFrame({'x':[1,2,3,4]})
        df_transformed_correct=pd.DataFrame({'x':[0,0,1,1]})
        bucket = Bucketizer(bins=2, bin_numeric=True)
        df_transformed = bucket.fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_no_replace(self):

        """
        transformation should be performed on y
        """
        df=pd.DataFrame({'x':[1,2,3]})
        df_transformed_correct=pd.DataFrame({'x':[1,2,3], 'x_binned':[0,0,0]})
        bucket = Bucketizer(bins=1, replace=False, bin_numeric=True)
        df_transformed = bucket.fit_transform(df)
        self.compare_DataFrame(df_transformed, df_transformed_correct)


    def test_empty_init(self):

        """
        init object with default parameters
        """

        bucket = Bucketizer()
        assert bucket.features==[]
        assert bucket.bins==2
        assert bucket.replace
        assert not bucket.bin_numeric
        assert not bucket.prefix


if __name__ == '__main__':
    unittest.main()