import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import evaluation
from datetime import datetime
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

__author__ = 'avrahami'


class ContentBasedRecommender(object):
    '''
    a class to implement Content based recommendation system, using TF-IDF idea between items names

    Parameters
    ----------
    gamma: float, default: 0.5
        the weight given to the tf-idf similarity, the (1-gamma) weight will be given to a general average rank value
        (as this number is closer to 1, we more 'trust' the TF-IDF idea and and less the average rank value)

    Attributes
    -----------
    _users_avg: dictionary
        average rank of each user organized as a dictionary (key: UserID, value: avg(rank)
    _items_avg: dictionary
        average rank of each item organized as a dictionary (key: ItemID, value: avg(rank)
    _user_item_avg: float
        the total average of all user-items in the dataset
    _similarities: data-frame
        matrix holding the similarities between any two items (based on the TF-IDF idea and the cosine similarity)
    '''

    def __init__(self, gamma=0.5):
        self.gamma = gamma
        self._users_avg = None
        self._items_avg = None
        self._user_item_avg = None
        self._similarities = None

    def train_tf_idf(self, path):
        '''
        training a content-base model based on the TF-IDF idea - this is the main function of this class.
        The idea here is to calculate TF-IDF of any item name, afterwards calculating the similarities between items
        based on this measure (cosine_similarities)

        :param path: location of the 'item' file description (this is the file containing name to any itemID)
        :return: nothing all is saved in the class object
        '''
        start_time = datetime.now()
        # reading the file with the items names
        items_data = pd.read_table(filepath_or_buffer=path, sep='::',
                               header=None, names=['itemid', 'name'], engine='python')
        # calculating the TF-IDF to each item
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(items_data['name'])

        # calculating the cosine similarity between any two itesm and holding it as a matrix
        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
        self._similarities = cosine_similarities
        duration = (datetime.now() - start_time).seconds
        print "CB model finished, took us {} minutes." \
              " Evaluation measures can be calculated only after predation step of this model".format(duration / 60.0)

    def _provide_pred(self, item_id, item_avg, train_data):
        '''
        internal function which yields a prediction value to a single row (that contains UserID, ItemID).
        The predicted value is according to the CB model, and returns average value in case of new user/item/both

        :param item_id: the itemID to predict the value
        :param item_avg: rank average of the current item we deal with
        :param train_data: data-frame containing user/item/ranks of the specific user we deal with. This is actually
                            a subset of the big train dataset, with only the relevant rows of the user we deal with
        :return: the predicted value
        '''

        # case we deal with a new user (have no history about him)
        if train_data.shape == 0:
            return item_avg
        # case we have only one rank of an item for this user
        if train_data.shape[0] == 1:
            relevant_similarities = self._similarities[item_id, int(train_data["ItemID"])]
            relevant_similarities_sum = relevant_similarities
        # case we have few items ranked by the current user
        else:
            relevant_similarities = self._similarities[item_id, train_data["ItemID"]]
            relevant_similarities_sum = sum(relevant_similarities)
        relevant_ranks = np.array(train_data["Rank"])

        # doing the calculations to create the prediction (this is a combined averages of the ranks according to the
        # similarity values (in case similarities do not sum to zero)
        if relevant_similarities_sum:
            similarity_based_pred = sum(relevant_similarities * relevant_ranks) / relevant_similarities_sum
            result = similarity_based_pred if item_avg is None else self.gamma*similarity_based_pred + (1-self.gamma)*item_avg
            return result
        # case similarities sum are zero, we have no other option but to return the average of the item
        else:
            return item_avg

    def predict(self, new_data, train_data):
        '''
        predict ranks to a new dataset of user/item. Mainly using the '_provide_pred' internal function which does
        the actual prediction per row

        :param new_data: data-frame to predict. Using the CB model we built in order to predict new data
        :param train_data: the training data-frame. Should hold 3 columns - UserID, ItemID and Rank. We need it also
                            along the prediction phase (usually it is not needed in this phase) because this kind of a
                            'lazy' model (same as KNN) which needs the ranks of the train dataset along prediction phase
                            The learning phase don't use this data, only the item names
        :return: list of predictions
        '''

        # calculating important values and saving them in the object
        self._users = set(train_data["UserID"])
        self._items = set(train_data["ItemID"])
        self._users_avg = train_data.groupby("UserID")["Rank"].mean().to_dict()
        self._items_avg = train_data.groupby("ItemID")["Rank"].mean().to_dict()
        self._user_item_avg = np.average(train_data["Rank"])

        # calling our internal function which predicts a single row, which the relevant function params
        prediction = new_data.apply(lambda x:
                                    self._provide_pred(item_id=x["ItemID"],
                                                       item_avg=self._items_avg.get(x["ItemID"]),
                                                       train_data=train_data[train_data["UserID"] == x["UserID"]]),
                                    axis=1)
        prediction = list(prediction)
        # replacing nan values with the general average (nan means that the item didn't exists in train)
        prediction = [self._user_item_avg if np.isnan(pred) else pred for pred in prediction]
        return prediction
