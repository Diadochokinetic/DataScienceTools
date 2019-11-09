import pandas as pd
import unittest
from dstools.preprocessing.NumScaler import NumScaler

#x1 should be scaled and x2 not
#df = pd.DataFrame({'x1':[1,2,3,4],'x2':['a','b','c','d']})
#df_transformed = NumScaler().fit_transform(df)
#print(df_transformed)

class TestNumScaler(unittest.TestCase):

    def test_string_value(self):

        """
        String values should not be affected by the NumScaler.
        """
        df = pd.DataFrame({'int':[1,2,3,4],'string':['a','b','c','d']})
        df_transformed = NumScaler().fit_transform(df)
        for x, y in zip(df['string'],df_transformed['string']):
            self.assertEqual(x, y)

    
    def test_int_value(self):

        """
        Int values should be scaled correctly by the NumScaler. Due to readability the transformed values are rounded to 6 digits for the equality test.
        """
        df = pd.DataFrame({'int':[1,2,3,4],'string':['a','b','c','d']})
        df_transformed_correct = pd.DataFrame({'int':[-1.341641,-0.447214,0.447214,1.341641],'string':['a','b','c','d']})
        df_transformed = NumScaler().fit_transform(df)
        for x, y in zip(df_transformed_correct['int'],df_transformed['int']):
            self.assertEqual(x, round(y, 6))  


    def test_original_unchanged(self):

        """
        The numerical data in the original DataFrame should not be affected.
        """
        df = pd.DataFrame({'int':[1,2,3,4],'string':['a','b','c','d']})
        df_original = pd.DataFrame({'int':[1,2,3,4],'string':['a','b','c','d']})
        df_transformed = NumScaler().fit_transform(df)  
        for x, y in zip(df_original['int'],df['int']):
            self.assertEqual(x, y)     


if __name__ == '__main__':
    unittest.main()