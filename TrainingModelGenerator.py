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
import sys



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



if __name__ == '__main__':

    if (len(sys.argv)) < 2:
        print(
            'Wrong number of arguments. Usage:\n BestConfig.py <INPUT FILE> ')
    else:
        training_file = (sys.argv[1])


    newsgroups_train = datasets.load_files(training_file,
                                           encoding='ISO-8859-1')



    remove_headers(newsgroups_train.data)

    stem_porter(newsgroups_train.data)



    #Chose f_classif over regression as it works better for this classfication problem
    anova_filter = SelectPercentile(f_classif)

    #Frequency vectorizer
    #Logarithmic value
    #Threshold to consider for df values

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, smooth_idf=False, sublinear_tf=True, max_df=0.87,
                                 min_df=0.0000001, binary=True)
    bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words="english", lowercase=True, analyzer='word',
                                        smooth_idf=False, sublinear_tf=True, max_df=0.95, min_df=2)


    vectors = vectorizer.fit_transform(newsgroups_train.data)


    svm_classifier= svm.LinearSVC(dual=False,C=1.22,tol=1e-5)

    anova_svm = Pipeline([('anova', anova_filter), ('svc', svm_classifier)])
    anova_svm.set_params(anova__percentile=91, svc__C=1.22,svc__tol=1e-5,svc__dual=False,svc__intercept_scaling=2.3).fit(vectors, newsgroups_train.target)
    #Persisting model and vectorizer

    joblib.dump(anova_svm, 'my_model.pkl', compress=9)
    joblib.dump(vectorizer,'my_vect.pkl',compress=9)


    pprint("Model generated and exported successfully->my_model.pkl")




