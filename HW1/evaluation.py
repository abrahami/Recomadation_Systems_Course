import pandas as pd
import numpy as np

__author__ = 'avrahami'


class Evaluation(object):
    '''
    a class to evaluate results, calculates the rmse and mse of any given two vectors

    Parameters
    ----------

    Attributes
    -----------
    rmse: float
        root mean square error
    mse: float
        mean absolute error
    '''

    def __init__(self):
        self.rmse = None
        self.mae = None

    def evaluate(self, true_ranks, predicted_ranks):
        '''
        the actual evaluation process

        Parameters
        ----------
        true_ranks: list
            the real rank values of each user with each item. Correctly it is given as a vector (1-dim), if needed to be
            changed in the future, we can convert is to a matrix
        predicted_ranks: list
            the predicted rank values of each user with each item. Correctly it is given as a vector (1-dim), if needed
            to be changed in the future, we can convert is to a matrix

        Returns
        -------
        all is stored in the class parameters (nothing is currently returned by the function
        '''

        if not isinstance(true_ranks, list) or not isinstance(predicted_ranks, list):
            print "each of the vectors in the evaluation function should be a list"
            raise Exception("Illegal input to the evaluation function")

        if len(true_ranks) != len(predicted_ranks):
            print "length of the two vectors must be the same, please check again"
            raise Exception("Illegal length of vectors")
        # convert both list to numpy array (if they are not so already) - it will help us with all calculations
        true_ranks = np.array(true_ranks)
        predicted_ranks = np.array(predicted_ranks)

        N = len(true_ranks)
        # calculating all the importatn measures we care about
        avg_power_error = np.sum(np.power((true_ranks-predicted_ranks), 2))*1.0/N
        avg_abs_error = np.sum(np.abs(true_ranks - predicted_ranks))*1.0/N
        self.rmse = np.sqrt(avg_power_error)
        self.mae = avg_abs_error
