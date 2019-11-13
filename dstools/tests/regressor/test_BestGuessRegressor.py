import pandas as pd
import numpy as np
import unittest
from dstools.regressor.BestGuessRegressor import BestGuessRegressor

class TestBestGuessRegressor(unittest.TestCase):

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


    def test_numeric_independent_numeric_dependent_best_guess(self):

        """
        a constant numeric class should be returned
        """

        df=pd.DataFrame({'x':[1,2,3],'y':[1,2,3]})
        y_predicted_correct=pd.DataFrame({'y_predicted':[2,2,2]})
        bgc = BestGuessRegressor()
        bgc.fit(df['x'],df['y'])
        y_predicted=pd.DataFrame(data=bgc.predict(df['x']),columns=['y_predicted'])
        self.compare_DataFrame(y_predicted,y_predicted_correct)


    def test_string_independent_numeric_dependent_best_guess(self):

        """
        a constant numeric class should be returned
        """

        df=pd.DataFrame({'x':['1','2','3'],'y':[1,2,3]})
        y_predicted_correct=pd.DataFrame({'y_predicted':[2,2,2]})
        bgc = BestGuessRegressor()
        bgc.fit(df['x'],df['y'])
        y_predicted=pd.DataFrame(data=bgc.predict(df['x']),columns=['y_predicted'])
        self.compare_DataFrame(y_predicted,y_predicted_correct)


if __name__ == '__main__':
    unittest.main()