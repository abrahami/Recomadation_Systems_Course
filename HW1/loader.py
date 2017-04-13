import pandas as pd

__author__ = 'avrahami'


class Loader(object):

    '''
    a class to load dataset for HW1 in Recom. systems course. Dataset are from 'Yelp' as given with the HW

    Parameters
    ----------
    path: string
        the location of the files in the local computer. should be the directory where all categories exists.
        path example: "C:\\Users\\abrahami\\Documents\\Private\\Uni\\BGU\\Reco.Systems - Bracha\\HW1\\Yelp"
    '''

    def __init__(self, path):
        # currently 'beautysvc' is problematic, should be fixed and then will be added
        self.valid_categories = ['active', 'homeservices', 'hotelstravel',
                                 'nightlife', 'pets', 'restaurants', 'shopping']
        # self.valid_categories = ['pets', 'hotelstravel']
        self.path = path

    def __load_single_category(self, train_location, test_location, factor=0):
        '''
        internal function of the base class which loads a specific train/test category from a specific file given

        Parameters
        ----------
        train_location: str
            exact location of the train dataset to load. This string includes the exact location of the file
        test_location: str
            exact location of the train dataset to load. This string includes the exact location of the file

        Returns
        -------
        list containing two datasets (train and test)
        '''
        train_data = pd.read_table(filepath_or_buffer=train_location, sep='::',
                                   header=None, names=['UserID', 'ItemID', 'Rank'], engine='python')
        train_data["UserID"] += factor
        train_data["ItemID"] += factor
        test_data = pd.read_table(filepath_or_buffer=test_location, sep='::',
                                   header=None, names=['UserID', 'ItemID', 'Rank'], engine='python')
        test_data["UserID"] += factor
        test_data["ItemID"] += factor

        return {'train_data': train_data, 'test_data': test_data}

    def load(self, category=None, d_train_all_categories=False):
        '''
        main function of this class - loads a single category datasets of all the categories possible to load
        (it combines all of the categories in such case into one big train/test dataset)

        Parameters
        ----------
        category: str, defaule: None
            the category to load (valid values are are one out of: 'active', 'beautysvc', 'homeservices',
            'hotelstravel','nightlife', 'pets', 'restaurants', 'shopping'.
            If given as 'None', the 'd_train_all_categories' must be True (otherwise it is not clear what should be
            loaded)
        d_train_all_categories: boolean, defaule: False
            boolean value indicating if all the datasets should be loaded together. If it is set to 'True', all datasets
            will be combined into one train/test dataset

        '''
        # Handling cases with problematic input by the user
        if category is not None and category not in self.valid_categories:
            print "Category you provided is not in the ones we know how to handle, please try again. Sorry"
            return 1
        if category is None and d_train_all_categories==False:
            print "You must specify one category OR set 'd_train_all_categories' to be True in order to load something"
            return 1
        # case input is OK, we'll check if we need to import one category or all of them
        if d_train_all_categories is False:
            # handling some typo problem in two cases ('training' and 'traning' are being confues in the file names)
            if category in ['hotelstravel', 'pets']:
                train_location = self.path + "\\" + category + "\\" + category + "_traning.txt"
            else:
                train_location = self.path + "\\" + category + "\\" + category + "_training.txt"
            # test file name is OK over all categories
            test_location = self.path+"\\"+category + "\\"+category+"_test.txt"
            return self.__load_single_category(train_location=train_location, test_location=test_location)
        else:
            train_data = pd.DataFrame(columns=['UserID', 'ItemID', 'Rank'],
                                      dtype=int)
            test_data = pd.DataFrame(columns=['UserID', 'ItemID', 'Rank'],
                                     dtype=int)
            for cat in self.valid_categories:
                # handling some typo problem in two cases ('training' and 'traning' are being confues in the file names)
                if cat in ['hotelstravel', 'pets']:
                    train_location = self.path + "\\" + cat + "\\" + cat + "_traning.txt"
                else:
                    train_location = self.path + "\\" + cat + "\\" + cat + "_training.txt"
                # test file name is OK over all categories
                test_location = self.path + "\\" + cat + "\\" + cat + "_test.txt"

                factor = (self.valid_categories.index(cat)+1)*100000
                cur_data = self.__load_single_category(train_location=train_location,
                                                       test_location=test_location,
                                                       factor=factor)
                train_data = pd.concat([train_data, cur_data['train_data']], ignore_index=True)
                test_data = pd.concat([test_data, cur_data['test_data']], ignore_index=True)
            return {'train_data': train_data.astype(int), 'test_data': test_data.astype(int)}





