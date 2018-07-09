# 20Newsgroupsprac


import numpy as np
import math


from sklearn.datasets import fetch_20newsgroups
cat = ['alt.atheism',
 'rec.autos',
 'sci.electronics',
'talk.politics.guns',
'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset="train",categories = cat)
newsgroups_test = fetch_20newsgroups(subset="test", categories =cat)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 10000)
vectors = vectorizer.fit_transform(newsgroups_train.data)
print(vectors.shape)
testvectors = vectorizer.fit_transform(newsgroups_test.data)
print(testvectors.shape)
#print(vectors.shape)
#print(testvectors.shape)

#Xtrain = newsgroups_train.data
Ytrain = newsgroups_train.target

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(150,))
clf.fit(vectors,Ytrain)

print(clf.predict(testvectors[355]))
print(newsgroups_test.target[355])
from sklearn.metrics import accuracy_score
print(accuracy_score(newsgroups_train.target,clf.predict(vectors)))
print(accuracy_score(newsgroups_test.target,clf.predict(testvectors)))

