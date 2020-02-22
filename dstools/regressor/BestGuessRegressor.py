import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error


class BestGuessRegressor(BaseEstimator):
    
    
    """
    The BestGuessRegressor creates a constant best guess for a given metric, e.g. mean absolute error. This regressor is for interval scaled numeric dependent variables. 
    
    Parameters
    ----------
    
    metric: function(y_true, y_predicted), default=sklearn.metrics.mean_absolute_error
        The best guess ist optmized on the given metric, either minimum or maximum.
        
    argmin: bool, default=True
        The given metric is either optimized on its minimum or maximum.
        
    n_splits: int, default=1000
        The value range of y will be splittet n_splits times. These n_splits + 1 values will be used for the argmin function.
        
    balance_mean: bool, default=False
        Sometimes an estimation needs to be very close to the mean of the real values. This requirement can be forced by adding an absolute error to the argmin/argmax function.


    Functions
    ---------

        __str__
            returns the name of the classifier as string

        fit(X, y)
            fit the classifier on the argmin or argmax of a given metric (default=sklearn.metrics.mean_absolute_error)

        predict(X, y=None)
            returns an pandas.Series with the constant best guess
        
        mean_diff(v1, v2, balance_mean)
            returns an error term if the constant best guess needs to be near mean


    Examples
    --------
    
    The BestGuessRegressor is insensitive to X values
    >>> y=[1,2,3]
    >>> y_pred = BestGuessRegressor().predict(X)
    >>> y_pred
    [2,2,2]
    """
    
    def __init__(self, metric=mean_absolute_error, argmin=True, n_splits=1000, balance_mean=False):
        
        self.best_guess_ = 42 #the answer to everyhing
        self.metric = metric
        self.argmin = argmin
        self.n_splits = n_splits
        self.balance_mean = balance_mean
  
        
    def __str__(self):
        
        return 'BestGuessRegressor'
    
        
    def fit(self, X, y):

        """
        fit the dstools.regressor.BestGuessRegressor by using the argmin or argmax of a given metric (default=sklearn.metrics.mean_absolute_error)

        Parameters
        ----------

            X: pandas.DataFrame
                independent variables

            y: pandas.Series, default=Nones
                dependent variable


        Output
        ------

            None

        """
        
        # create range of arguments to be used for optimization
        args = [min(y) + (max(y) - min(y)) * i / self.n_splits for i in range(0,self.n_splits + 1)]
        
        # get either minimum or maximum of a metric score to optimize the best guess
        if self.argmin:
            
            self.best_guess_ = args[np.argmin([self.metric(y,np.full(len(y),x) + self.mean_diff(np.mean(y), x, self.balance_mean)) for x in args])]
        
        else:
            
            self.best_guess_ = args[np.argmax([self.metric(y,np.full(len(y),x) + self.mean_diff(np.mean(y), x, self.balance_mean)) for x in args])]    
            
    
    def mean_diff(self, v1, v2, balance_mean):

        """
        creates an error term in case the constant guess needs to be near men

        Parameters
        ----------

            v1: int
                value 1

            v2: int
                value 2

            balance_mean: boolean
                if True return diff, else return 0

        
        Output
        ------

            int

        """
        
        if balance_mean:
            
            return np.abs(v1-v2)
        
        else:
            
            return 0
    
    
    def predict(self, X, y=None):
        
        """
        returns a constant best guess following the given metric (default=sklearn.metrics.mean_absolute_error)

        Parameters
        ----------

            X: pandas.DataFrame
                independent variables

            y: pandas.Series, default=None
                dependent variable, not needed in predict


        Output
        ------

            pandas.Series containing a constant best guess for the dependent variable

        """

        y_predicted = pd.Series(np.full(X.shape[0], self.best_guess_))
        return y_predicted
