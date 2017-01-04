from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn import linear_model
import nltk
from sklearn import svm
from sklearn import ensemble
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, SelectPercentile
from nltk.stem.porter import *
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


def remove_headers(data):
    for i, text in enumerate(data):
        _before, _blankline, after = text.partition('\n\n')
        data[i] = after


def stem_porter(data):
    stemmer = PorterStemmer()
    stem_text = []
    for i, text in enumerate(data):
        stem_text = text.split()
        singles = [stemmer.stem(word) for word in stem_text]
        data[i] = (' ').join(singles)


newsgroups_test = datasets.load_files("dataset/Test",
                                      encoding='ISO-8859-1')
remove_headers(newsgroups_test.data)
stem_porter(newsgroups_test.data)


vectorizer = joblib.load('my_vect.pkl')
vectors_test = vectorizer.transform(newsgroups_test.data)
anova_svm=joblib.load('my_model.pkl')

pprint("Model and vectoizer loaded")

pred_svm = anova_svm.predict(vectors_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Unigram Linear SVM======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)

