import pandas as pd
import numpy as np
import unittest
from dstools.preprocessing.FeatureConverter import FeatureConverter

class TestFeatureConverter(unittest.TestCase):

    """
    =======================
    === type conversion ===
    =======================
    """
    def test_int_to_string(self):

        """
        int values should be converted to string
        """
        df=pd.DataFrame({'int':[1,2,3,4]})
        df_transformed_correct=pd.DataFrame({'int':['1','2','3','4']})
        df_transformed = FeatureConverter(columns_to_string='int').fit_transform(df)
        for x, y in zip(df_transformed['int'], df_transformed_correct['int']):
            self.assertEqual(x, y)

    
    def test_float_to_string(self):

        """
        int values should be converted to string
        """
        df=pd.DataFrame({'float':[1.2,2.2,3.2,4.2]})
        df_transformed_correct=pd.DataFrame({'float':['1.2','2.2','3.2','4.2']})
        df_transformed = FeatureConverter(columns_to_string='float').fit_transform(df)
        for x, y in zip(df_transformed['float'], df_transformed_correct['float']):
            self.assertEqual(x, y)


    def test_string_to_int(self):

        """
        string values should be converted to int
        """
        df=pd.DataFrame({'string':['1','2','3','4']})
        df_transformed_correct=pd.DataFrame({'string':[1,2,3,4]})
        df_transformed = FeatureConverter(columns_to_int='string').fit_transform(df)
        for x, y in zip(df_transformed['string'], df_transformed_correct['string']):
            self.assertEqual(x, y)


    def test_float_to_int(self):

        """
        float values should be converted to int
        """
        df=pd.DataFrame({'float':[1.2,2.2,2.9,4.3]})
        df_transformed_correct=pd.DataFrame({'float':[1,2,3,4]})
        df_transformed = FeatureConverter(columns_to_int='float').fit_transform(df)
        for x, y in zip(df_transformed['float'], df_transformed_correct['float']):
            self.assertEqual(x, y)


    def test_string_to_float(self):

        """
        string values should be converted to float
        """
        df=pd.DataFrame({'string':['1.2','2.2','3.2','4.2']})
        df_transformed_correct=pd.DataFrame({'string':[1.2,2.2,3.2,4.2]})
        df_transformed = FeatureConverter(columns_to_float='string').fit_transform(df)
        for x, y in zip(df_transformed['string'], df_transformed_correct['string']):
            self.assertEqual(x, y)


    def test_int_to_float(self):

        """
        int values should be converted to float
        """
        df=pd.DataFrame({'int':[1,2,3,4]})
        df_transformed_correct=pd.DataFrame({'int':[1.0,2.0,3.0,4.0]})
        df_transformed = FeatureConverter(columns_to_float='int').fit_transform(df)
        for x, y in zip(df_transformed['int'], df_transformed_correct['int']):
            self.assertEqual(x, y)

    
    """
    ===================
    === replacement ===
    ===================
    """
    def test_replace_one_value(self):

        """
        single value should be replaced
        """
        df=pd.DataFrame({'int':[1,2,3,4]})
        df_transformed_correct = pd.DataFrame({'int':[42,2,3,4]})
        df_transformed = FeatureConverter(columns_with_replace={'int':{1:42}}).fit_transform(df)
        for x, y in zip(df_transformed['int'], df_transformed_correct['int']):
            self.assertEqual(x, y)

    
    def test_replace_two_values(self):

        """
        two values should be replaced
        """
        df=pd.DataFrame({'string':['1','2','3','4']})
        df_transformed_correct = pd.DataFrame({'string':['hello','world','3','4']})
        df_transformed = FeatureConverter(columns_with_replace={'string':{'1':'hello','2':'world'}}).fit_transform(df)
        for x, y in zip(df_transformed['string'], df_transformed_correct['string']):
            self.assertEqual(x, y)


    def test_replace_two_columns(self):

        """
        values in multiple columns should be replaced
        """
        df=pd.DataFrame({'int':[1,2,3,4], 'string':['1','2','3','4']})
        df_transformed_correct = pd.DataFrame({'int':[42,2,3,4], 'string':['hello','world','3','4']})
        df_transformed = FeatureConverter(columns_with_replace={'int':{1:42}, 'string':{'1':'hello','2':'world'}}).fit_transform(df)
        for x, y in zip(df_transformed['string'], df_transformed_correct['string']):
            self.assertEqual(x, y)
        for x, y in zip(df_transformed['int'], df_transformed_correct['int']):
            self.assertEqual(x, y)


    """
    ====================
    === create flags ===
    ====================
    """
    def test_one_flag(self):

        """
        one flag should be created
        """
        df=pd.DataFrame({'int':[1,2,3,4]})
        df_transformed_correct=pd.DataFrame({'int':[1,2,3,4], 'int_1':[1,0,0,0]})
        df_transformed=FeatureConverter(value_flags={'int':[1]}).fit_transform(df)
        for x, y in zip(df_transformed['int_1'], df_transformed_correct['int_1']):
            self.assertEqual(x, y)


    def test_two_flags(self):

        """
        two flags should be created
        """
        df=pd.DataFrame({'int':[1,2,3,4]})
        df_transformed_correct=pd.DataFrame({'int':[1,2,3,4], 'int_1':[1,0,0,0], 'int_2':[0,1,0,0]})
        df_transformed=FeatureConverter(value_flags={'int':[1,2]}).fit_transform(df)
        for x, y in zip(df_transformed['int_1'], df_transformed_correct['int_1']):
            self.assertEqual(x, y)
        for x, y in zip(df_transformed['int_2'], df_transformed_correct['int_2']):
            self.assertEqual(x, y)


    def test_multiple_flags(self):

        """
        multiple flags should be created
        """
        df=pd.DataFrame({'int':[1,2,3,4], 'string':['1','2','3','4']})
        df_transformed_correct=pd.DataFrame({'int':[1,2,3,4], 'string':['1','2','3','4'], 'int_1':[1,0,0,0], 'int_2':[0,1,0,0], 'string_1':[1,0,0,0]})
        df_transformed=FeatureConverter(value_flags={'int':[1,2], 'string':['1']}).fit_transform(df)
        for x, y in zip(df_transformed['int_1'], df_transformed_correct['int_1']):
            self.assertEqual(x, y)
        for x, y in zip(df_transformed['int_2'], df_transformed_correct['int_2']):
            self.assertEqual(x, y)
        for x, y in zip(df_transformed['string_1'], df_transformed_correct['string_1']):
            self.assertEqual(x, y)


    """
    ===================
    === drop column ===
    ===================
    """
    def test_drop_one(self):

        """
        one column should be dropped
        """
        df=pd.DataFrame({'string1':['1','2','3','4'],'string2':['1','2','3','4'],'int':[1,2,3,4]})
        df_transformed_correct=pd.DataFrame({'string2':['1','2','3','4'],'int':[1,2,3,4]})
        df_transformed=FeatureConverter(columns_to_drop='string1').fit_transform(df)
        for x, y in zip(df_transformed.columns, df_transformed_correct.columns):
            self.assertAlmostEqual(x, y)


    def test_drop_two(self):

        """
        two columns should be dropped
        """
        df=pd.DataFrame({'string1':['1','2','3','4'],'string2':['1','2','3','4'],'int':[1,2,3,4]})
        df_transformed_correct=pd.DataFrame({'string2':['1','2','3','4']})
        df_transformed=FeatureConverter(columns_to_drop=['string1','int']).fit_transform(df)
        for x, y in zip(df_transformed.columns, df_transformed_correct.columns):
            self.assertAlmostEqual(x, y)


if __name__ == '__main__':
    unittest.main()