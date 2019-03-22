import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
import os


# Helper class for data cleaning/preparation
class Helper:
    def __init__(self, path = "datasets"):
        self.path = path
        self.train_path = os.path.join(path, "train.csv")
        self.test_path = os.path.join(path, "test.csv")
        self.count_vect = self.build_vocabulary()

    def build_vocabulary(self):
        # Read the datasets
        train_data = pd.read_csv(self.train_path, sep = "~")
        #print(train_data.info())
        test_data = pd.read_csv(self.test_path, sep = "~")
        #print(test_data.info())
        self.train_desc = train_data["Description"]
        self.test_desc = test_data["Description"]
        self.target = list(train_data["Is_Response"])
        corpus = np.concatenate((self.train_desc.values, self.test_desc.values))
        count_vect = CountVectorizer(stop_words = 'english', min_df = 5, ngram_range = (2, 2))
        count_vect.fit(corpus)
        return count_vect

    def get_target(self):
        return self.target

    def get_train_desc(self):
        return self.train_desc

    def get_test_desc(self):
        return self.test_desc

    def get_train_data(self):
        return self.count_vect.transform(self.train_desc.values), self.target

    def get_test_data(self):
        return self.count_vect.transform(self.test_desc.values)

    def get_feature_names(self):
        return self.count_vect.get_feature_names()

    def generate_result(self, pred, suffix = None):
        test_data = pd.read_csv(self.test_path, sep = "~")
        file_name = "submission.csv"
        if suffix is not None:
            file_name = "submission_{}.csv".format(suffix)
        with open(file_name, "w") as f:
            f.write("User_ID~Is_Response\n")
            for i, target in enumerate(pred):
                f.write("{}~{}\n".format(test_data["User_ID"][i], target))
                #print("{} : {} -> {}".format(test_data["User_ID"][i], target, test_data["Description"][i]))
                #input()

    def clean_description(self, doc):
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = [token.translate(table) for token in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]

        return tokens

