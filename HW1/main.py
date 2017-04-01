import loader

# example of loading one specific category (here it is 'pets')
my_path = "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\Reco.Systems - Bracha\\HW1\\Yelp"
loader_obj = loader.Loader(path=my_path)
data = loader_obj.load(category='pets', d_train_all_categories=False)
train = data['train_data']
test = data['test_data']

print "Loading has just finished, shape of the train dataset is {}," \
      " shape of the test dataset is {}".format(train.shape, test.shape)


# of loading all categories
my_path = "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\Reco.Systems - Bracha\\HW1\\Yelp"
loader_obj = loader.Loader(path=my_path)
data = loader_obj.load(d_train_all_categories=True)
train = data['train_data']
test = data['test_data']

print "Loading has just finished, shape of the train dataset is {}," \
      " shape of the test dataset is {}".format(train.shape, test.shape)