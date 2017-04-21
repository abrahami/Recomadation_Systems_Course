import pandas as pd
import numpy as np
import evaluation
import recommender_system
import content_based
import scipy as sc

__author__ = 'avrahami'


class HybridModeling(object):
    '''
    a class to implement a hybrid model which is baed on the SVD and the CB models

    Parameters
    ----------
    gamma: float, default: 0.5
        paramter relevant to the CB model.
        This is the weight given to the tf-idf similarity, the (1-gamma) weight will be given to a general average rank
        (as this number is closer to 1, we more 'trust' the TF-IDF idea and and less the average rank value)
    svd_latent_size: integer, default: 5
        number of latent feature to create in the SVD model. Only integer positive values are allowed
    svd_iterations: maximum number of iteration to run in the SVD model. The algo can stop before
                    (if the RMSE rises between iterations). Default=100. Must be positive integer
    Attributes
    -----------
    _svd_model: SVDRecommender class object
        the model which is built along the algorithm, saved as an object of the class (so it can be used later mainly
        for new prediction purposes)
    _cb_model: ContentBasedRecommender class object
        the model which is built along the algorithm, saved as an object of the class (so it can be used later mainly
        for new prediction purposes)
    '''

    def __init__(self, gamma=0.5, svd_latent_size=5, svd_iterations=100):
        self.gamma = gamma
        self.svd_latent_size = svd_latent_size
        self.svd_iterations = svd_iterations
        self._svd_model = None
        self._cb_model = None

    def train_hybrid(self, train_data, items_path):
        '''
        training a hybrid model based the concept of building two models and giving same weight to each one.
        This mean that final prediction here will be 50% based on the SVD and 50% based on the CB model

        :param train_data: he data-frame to learn from. Should hold 3 columns - UserID, ItemID and Rank
        :param items_path: location of the 'item' file description (this is the file containing name to any itemID) -
                            this is used inside the CB model
        :return: nothing all is saved in the class object
        '''

        # svd model
        svd_obj = recommender_system.SVDRecommender()
        svd_obj.train_base_model(data=train_data, latent_size=self.svd_latent_size, iterations=self.svd_iterations)
        self._svd_model = svd_obj
        # CB model
        cb_obj = content_based.ContentBasedRecommender(gamma=self.gamma)
        cb_obj.train_tf_idf(path=items_path)
        self._cb_model = cb_obj

    def predict(self, new_data, train_data):
        '''
        predict ranks to a new dataset of user/item. It calls each algorithm (SVD and CB ones) and then mixed both
        predictions together in a same weight manner

        :param new_data: data-frame to predict
        :param train_data: the training data-frame. Should hold 3 columns - UserID, ItemID and Rank. We need it also
                            along the prediction phase (usually it is not needed in this phase) because the CB is kind
                            of a 'lazy' model (same as KNN) which needs the ranks of the train dataset along prediction
                             phase. The learning phase don't use this data in the CB model, only the item names
        :return: list of predictions
        '''

        svd_prediction = self._svd_model.predict(new_data=new_data)
        cb_prediction = self._cb_model.predict(new_data=new_data, train_data=train_data)

        # calculating the pearson correlation measure between the two - it is interesting to see, since we wish these
        # two vectors not to have too high correlation (if they do have - the potential of the hybrid is not too high)
        print "\nPearson correlation between" \
              " the two predictions is {}".format(sc.stats.pearsonr(svd_prediction, cb_prediction)[0])
        # returns the average between the two predictions
        return [np.mean(i) for i in zip(svd_prediction, cb_prediction)]