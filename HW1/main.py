import loader
import evaluation
import svd
import content_based
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import hybrid

__author__ = 'avrahami'

''' example of evaluation process
true_vec = [1, 2, 3, 4, 5]
pred_vec = [3, 3, 3, 3, 3]
eval_obj = evaluation.Evaluation()
eval_obj.evaluate(true_ranks=true_vec, predicted_ranks=pred_vec)
print "Evaluation has just ended, the RMSE is {}, the MAE is {}".format(eval_obj.rmse, eval_obj.mae)
'''

# region 1. Configurations (USER MUST DEFINE THESE)
# must be out of the following: active, homeservices, hotelstravel, nightlife, pets, restaurants, shopping, all
cur_category = 'all'
data_path = "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\Reco.Systems - Bracha\\HW1\\Yelp"
model = 'svd'  # should be one out of svd/cb/hybrid
# endregion

# region 2. Data load
loader_obj = loader.Loader(path=data_path)
if cur_category!='all':
    data = loader_obj.load(category=cur_category, d_train_all_categories=False)
else:
    data = loader_obj.load(d_train_all_categories=True)

train = data['train_data']
test = data['test_data']
print "Loading has just finished, shape of the train dataset is {}," \
      " shape of the test dataset is {}.\n" \
      "There are {} distinct users and {} distinct items in the training dataset." \
      " There are {} distinct users and {} distinct items in the test dataset\n".\
    format(train.shape, test.shape, len(train["UserID"].unique()), len(train["ItemID"].unique()),
           len(test["UserID"].unique()), len(test["ItemID"].unique()))
# endregion

# region 3. Modeling
if model == 'svd':
    svd_obj = svd.SVDRecommender()
    svd_obj.train_base_model(data=train, latent_size=5, iterations=10, gamma=0.01, lamda=0.01)
    test_prediction = svd_obj.predict(new_data=test)

if model == 'cb':
    cb_obj = content_based.ContentBasedRecommender(gamma=0.5)
    items_path = data_path + "\\"+cur_category + "\\"+"items.txt"
    cb_obj.train_tf_idf(path=items_path)
    test_prediction = cb_obj.predict(new_data=test, train_data=train)

if model == 'hybrid':
    hybrid_obj = hybrid.HybridModeling()
    items_path = data_path + "\\"+cur_category + "\\"+"items.txt"
    hybrid_obj.train_hybrid(train_data=train, items_path=items_path)
    test_prediction = hybrid_obj.predict(new_data=test, train_data=train)
# endregion

# region 4. Evaluation
eval_obj = evaluation.Evaluation()
eval_obj.evaluate(true_ranks=list(test["Rank"]), predicted_ranks=test_prediction)
print "\n{} model results: the RMSE of the test data-set is {}, the MAE is {}". \
    format(model, eval_obj.rmse, eval_obj.mae)
# endregion
