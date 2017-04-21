import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import evaluation
from datetime import datetime

__author__ = 'avrahami'


class SVDRecommender(object):
    '''
    a class in order to build and run the SVD algorithm

    Parameters
    ----------
    nothing here, all are when running the algorithm in the 'train_base_model'

    Attributes
    -----------
    rmse: float
        root mean square error of the validation data (internal dataset we hold to test a current solution)
    _users_avg: dictionary
        average rank of each user organized as a dictionary (key: UserID, value: avg(rank)
    _items_avg: dictionary
        average rank of each item organized as a dictionary (key: ItemID, value: avg(rank)
    _user_item_avg: float
        the total average of all user-items in the dataset
    _average: float
        the average value of all user-items in the validation dataset
        (internal dataset we hold to test a current solution)
    _users: set
        all users appear in the train dataset
    _items: set
        all items appear in the train dataset
    _full_pred_matrix: data-frame
        matrix which holds the non empty predition of each user-item (after the matrix factorization phase). It is
        not the final prediction we output, since we reduce the bias terms and the average
    _b_u: data-frame
        bias value to each user stored as a df
    _b_i: data-frame
        bias value to each item stored as a df
    '''

    def __init__(self):
        self.rmse = np.inf
        self._users_avg = None
        self._items_avg = None
        self._user_item_avg = None
        self._average = None
        self._users = None
        self._items = None
        self._full_pred_matrix = None
        self._b_u = None
        self._b_i = None

    def _provide_pred(self, single_row):
        '''
        internal function which yields a prediction value to a single row (that contains UserID, ItemID).
        The predicted value is according to the SVD model, and returns average value in case of new user/item/both

        :param single_row: a row with userID and ItemID which needs to be predicted
        :return: the predicted value
        '''

        # creating indicators whether we deal with new user/item (or maybe both)
        existing_user = single_row.UserID in self._users
        existing_item = single_row.ItemID in self._items

        # case we have the userID and the itemID in our training set and we can pull the prediction easily
        if existing_user and existing_item:
            dot_prod = self._full_pred_matrix.loc[single_row.UserID, single_row.ItemID]
            return self._average + self._b_u.loc[single_row.UserID] + self._b_i.loc[single_row.ItemID] + dot_prod
        # case we only have the UserID in our training set (case of a new item to the system) - we'll return the average
        # of ranking for this user
        elif existing_user:
            return self._users_avg.get(single_row.UserID)
        # case we only have the ItemID in our training set (case of a new item to the system) - we'll return the average
        # of ranking for this user
        elif existing_item:
            return self._items_avg.get(single_row.ItemID)
        # case we deal with a new user and item - we'll return the average raking over all dataset
        else:
            return self._user_item_avg

    def train_base_model(self, data, latent_size=20, iterations=100, lamda=0.05, gamma=0.05, verbose=True):
        '''
        training a SVD model - this is the main function of this class. Doing the iteration of learning according
        to the gradient descent idea, as described in the exercise instructions

        :param data: the data-frame to learn from. Should hold 3 columns - UserID, ItemID and Rank
        :param latent_size: number of latent feature to create. Default=20. Must be positive integer
        :param iterations: maximum number of iteration to run. The algo can stop before (if the RMSE rises between
                iterations). Default=100. Must be positive integer
        :param lamda: the lambda parameter in the algorithm, controls the learning rate.
                Default=0.05. Must be positive number, smaller than 1
        :param gamma: the gamma parameter in the algorithm, controls the learning rate.
                Default=0.05. Must be positive number, smaller than 1
        :param verbose: Boolean, whether to print things along the run
        :return: nothing, all stored in the class object
        '''

        start_time = datetime.now()
        # splitting to train and validation (validation is used as an internal dataset to avoid overfitting)
        train_data, validation_data = train_test_split(data, test_size=0.3, random_state=42)

        # calculating important values and saving them in the object
        self._users = set(train_data["UserID"])
        self._items = set(train_data["ItemID"])
        self._users_avg = data.groupby("UserID")["Rank"].mean().to_dict()
        self._items_avg = data.groupby("ItemID")["Rank"].mean().to_dict()
        self._user_item_avg = np.average(data["Rank"])

        # setting random values to the models parameters
        b_u = np.random.uniform(-0.1, 0.1, size=len(self._users))
        b_i = np.random.uniform(-0.1, 0.1, size=len(self._items))
        p_u = np.random.uniform(-0.1, 0.1, size=(len(self._users), latent_size))
        q_i = np.random.uniform(-0.1, 0.1, size=(len(self._items), latent_size))
        avg = np.average(train_data["Rank"])
        self._average=avg

        # converting some to data-frames (easier to save and handle later)
        p_u_df = pd.DataFrame(p_u, index=list(self._users), columns=range(0, latent_size))
        q_i_df = pd.DataFrame(q_i, index=list(self._items), columns=range(0, latent_size))
        b_u_df = pd.DataFrame(b_u, index=list(self._users))
        b_i_df = pd.DataFrame(b_i, index=list(self._items))

        # building the evaluation object to be used inside the loop
        eval_obj = evaluation.Evaluation()
        # staritng the gradinet descent phase, the loop can stop before we reach 'iterations' number of cycles
        for i in xrange(iterations):
            # start_time = datetime.now()
            # looping over each row in the dataset (user/item/rank) and updating the params
            for index, row in train_data.iterrows():
                cur_user = row["UserID"]
                cur_item = row["ItemID"]
                cur_rank = row["Rank"]
                dot_prod = sum(p_u_df.loc[cur_user] * q_i_df.loc[cur_item])
                cur_error = float(cur_rank-avg-b_u_df.loc[cur_user]-b_i_df.loc[cur_item]-dot_prod)
                b_u_df.loc[cur_user] += lamda * (cur_error - gamma * b_u_df.loc[cur_user])
                b_i_df.loc[cur_item] += lamda * (cur_error - gamma * b_i_df.loc[cur_item])
                q_i_df.loc[cur_item] += lamda * (cur_error * p_u_df.loc[cur_user] - gamma * q_i_df.loc[cur_item])
                p_u_df.loc[cur_user] += lamda * (cur_error * q_i_df.loc[cur_item] - gamma * p_u_df.loc[cur_user])
            # end of inner loop, now calculating the errors and decide if to go for another loop
            self._full_pred_matrix = p_u_df.dot(q_i_df.transpose())
            self._b_i = b_i_df
            self._b_u = b_u_df
            validation_pred = self.predict(new_data=validation_data)
            eval_obj.evaluate(true_ranks=list(validation_data["Rank"]), predicted_ranks=validation_pred)
            # duration = (datetime.now() - start_time).seconds
            # print "Loop number {}, the RMSE is {}," \
            #       "this loop took us {} minutes".format(i, eval_obj.rmse, duration/60.0)
            # case the RMSE was improved, we'll save it and go for another loop
            if eval_obj.rmse < self.rmse:
                self.rmse = eval_obj.rmse
            # case we need to finish the algorithm, RMSE wasn't improved
            else:
                train_pred = self.predict(new_data=train_data)
                eval_obj.evaluate(true_ranks=list(train_data["Rank"]), predicted_ranks=train_pred)
                duration = (datetime.now() - start_time).seconds
                print "SVD model finished, took us {} loops and {} minutes." \
                      " RMSE measure in the train dataset is {}" \
                      " MAE measure in the train dataset is {}".format(i, duration/60.0, eval_obj.rmse, eval_obj.mae)
                break

    def predict(self, new_data):
        '''
        predict ranks to a new dataset of user/item. Mainly using the '_provide_pred' internal function which does
        the actual prediction per row

        :param new_data: data-frame to predict. Using the SVD model we built in order to predict new data
        :return: list of predictions
        '''

        # calling our internal function which predicts a single row
        prediction = new_data.apply(lambda x: self._provide_pred(single_row=x), axis=1)
        if isinstance(prediction, pd.DataFrame):
            return list(prediction[0])
        else:
            return list(prediction)
