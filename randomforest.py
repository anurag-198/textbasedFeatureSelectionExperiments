import pandas as pd
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

import warnings

old_stdout = sys.stdout
log_file = open("message.log","a+")
sys.stdout = log_file


### CONVERTING TEXT TO VECTORS

categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',categories=categories, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test',categories=categories, random_state=42)


vectorizer = TfidfVectorizer()
vectors_train_data = vectorizer.fit_transform(newsgroups_train.data)
vectors_test_data = vectorizer.fit_transform(newsgroups_test.data)

vectors_train_data.nnz / float(vectors_train_data.shape[0])
vectors_test_data.nnz / float (vectors_test_data.shape[0])


vectors_train_target = newsgroups_train.target
vectors_test_target = newsgroups_test.target

vectors_train_target = np.matrix([vectors_train_target])
vectors_test_target = np.matrix([vectors_test_target])

vectors_train_target = vectors_train_target.transpose()
vectors_test_target = vectors_test_target.transpose()
features = vectorizer.get_feature_names()

### RANDOM FOREST

rf = RandomForestClassifier(n_estimators=100, random_state = 42 ) # initialize
rf.fit(vectors_train_data, vectors_train_target)


featureImportance = rf.feature_importances_

n = len(featureImportance)
x = range(n)

ffi_pair = zip(features,featureImportance)

ffi_pair.sort(key = lambda x:x[1])

sol = ffi_pair[::-1]
print (sol[:100])

sys.stdout = old_stdout
log_file.close()