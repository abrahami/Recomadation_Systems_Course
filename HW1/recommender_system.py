import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
        self.rmse = None
        self.final_model = None


    def train_base_model(self, data, latent_size=20, iterations=100, lamda=0.05, gamma=0.05, verbose=True):
        train_data, validation_data = train_test_split(data, test_size=0.3, random_state=42)

        users = list(set(train_data.iloc[:,0]))
        items = list(set(train_data.iloc[:,1]))

        b_u = np.random.uniform(-0.1, 0.1, size=len(users))
        b_i = np.random.uniform(-0.1, 0.1, size=len(items))
        p_u = np.random.uniform(-0.1, 0.1, size=(len(users), latent_size))
        q_i = np.random.uniform(-0.1, 0.1, size=(len(items), latent_size))
        avg = np.average(data.iloc[:,2])

        p_u_df = pd.DataFrame(p_u, index=users, columns=range(0,latent_size))
        q_i_df = pd.DataFrame(q_i, index=items, columns=range(0,latent_size))
        b_u_df = pd.DataFrame(b_u, index=users)
        b_i_df = pd.DataFrame(b_i, index=items)
        for i in iterations:
            for index, row in train_data.iterrows():
                cur_user = row["UserID"]
                cur_item = row["ItemID"]
                cur_rank = row["ItemID"]
                dot_prod = sum(p_u_df.iloc[cur_user]*q_i_df.iloc[cur_item])
                cur_error = cur_rank-avg-b_u_df.iloc[cur_user]-b_i_df.iloc[cur_item]-dot_prod

