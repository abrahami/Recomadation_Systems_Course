import loader
import evaluation
import recommender_system
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# example of loading one specific category (here it is 'pets')
'''
my_path = "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\Reco.Systems - Bracha\\HW1\\Yelp"
loader_obj = loader.Loader(path=my_path)
data = loader_obj.load(category='pets', d_train_all_categories=False)
train = data['train_data']
test = data['test_data']
'''

# example of loading all categories
my_path = "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\Reco.Systems - Bracha\\HW1\\Yelp"
loader_obj = loader.Loader(path=my_path)
data = loader_obj.load(d_train_all_categories=True)
train = data['train_data']
test = data['test_data']


print "Loading has just finished, shape of the train dataset is {}," \
      " shape of the test dataset is {}.\n" \
      "There are {} distinct users and {} distinct items in the training dataset." \
      " There are {} distinct users and {} distinct items in the test dataset".\
    format(train.shape, test.shape, len(train["UserID"].unique()), len(train["ItemID"].unique()),
           len(test["UserID"].unique()), len(test["ItemID"].unique()))


# example of evaluation process
true_vec = [1, 2, 3, 4, 5]
pred_vec = [3, 3, 3, 3, 3]
eval_obj = evaluation.Evaluation()
eval_obj.evaluate(true_ranks=true_vec, predicted_ranks=pred_vec)
print "Evaluation has just ended, the RMSE is {}, the MAE is {}".format(eval_obj.rmse, eval_obj.mae)


# example of running a recommendation engine process
svd_obj = recommender_system.RecommenderSystem()
svd_obj.train_base_model(data=train, iterations=10)
