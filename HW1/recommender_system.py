import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import evaluation
from datetime import datetime

__author__ = 'avrahami'


class RecommenderSystem(object):
    '''
    a class to implement all the algorithms we develop for recommendations

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
        self.rmse = np.inf
        self.final_model = None
        self._average = None
        self._users = None
        self._items = None
        self._full_pred_matrix = None
        self._b_u = None
        self._b_i = None

    def _provide_pred(self, single_row, train_data):
        existing_user = single_row.UserID in self._users
        existing_item = single_row.ItemID in self._items
        # case we have the userID and the itemID in our training set and we can pull the prediction easily
        if existing_user and existing_item:
            dot_prod = self._full_pred_matrix.loc[single_row.UserID, single_row.ItemID]
            return self._average + self._b_u.loc[single_row.UserID] + self._b_i.loc[single_row.ItemID] + dot_prod
        # case we only have the UserID in our training set (case of a new item to the system) - we'll return the average
        # of ranking for this user
        elif existing_user:
            return train_data[train_data["UserID"] == single_row.UserID]["Rank"].mean()
        # case we only have the ItemID in our training set (case of a new item to the system) - we'll return the average
        # of ranking for this user
        elif existing_item:
            return train_data[train_data["ItemID"] == single_row.ItemID]["Rank"].mean()
        # case we deal with a new user and item - we'll return the average raking over all dataset
        else:
            return train_data["Rank"].mean()

    def train_base_model(self, data, latent_size=20, iterations=100, lamda=0.05, gamma=0.05, verbose=True):
        train_data, validation_data = train_test_split(data, test_size=0.3, random_state=42)

        self._users = set(train_data["UserID"])
        self._items = set(train_data["ItemID"])

        b_u = np.random.uniform(-0.1, 0.1, size=len(self._users))
        b_i = np.random.uniform(-0.1, 0.1, size=len(self._items))
        p_u = np.random.uniform(-0.1, 0.1, size=(len(self._users), latent_size))
        q_i = np.random.uniform(-0.1, 0.1, size=(len(self._items), latent_size))
        avg = np.average(train_data["Rank"])
        self._average=avg

        p_u_df = pd.DataFrame(p_u, index=list(self._users), columns=range(0, latent_size))
        q_i_df = pd.DataFrame(q_i, index=list(self._items), columns=range(0, latent_size))
        b_u_df = pd.DataFrame(b_u, index=list(self._users))
        b_i_df = pd.DataFrame(b_i, index=list(self._items))

        # building the evaluation object to be used inside the loop
        eval_obj = evaluation.Evaluation()
        for i in xrange(iterations):
            start_time = datetime.now()
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
            validation_pred = validation_data.apply(lambda x: self._provide_pred(single_row=x, train_data=train_data),
                                                         axis=1)
            if isinstance(validation_pred, pd.DataFrame):
                validation_pred=list(validation_pred[0])
            else:
                validation_pred = list(validation_pred)
            eval_obj.evaluate(true_ranks=list(validation_data["Rank"]), predicted_ranks=validation_pred)
            duration = (datetime.now() - start_time).seconds
            print "Loop number {}, the RMSE is {}, this loop took us {} minutes".format(i, eval_obj.rmse, duration/60.0)
            if eval_obj.rmse < self.rmse:
                self.rmse = eval_obj.rmse
            else:
                print "RMSE measure wasn't improved and got the value of {}, ending the algorithm".format(eval_obj.rmse)
                break










