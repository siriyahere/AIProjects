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

    if (len(sys.argv)) < 3:
        print(
            'Wrong number of arguments. Usage:\n BestConfig.py <INPUT FILE> ')
    else:
        training_file = (sys.argv[1])
        test_file = (sys.argv[2])

    newsgroups_train = datasets.load_files(training_file,
                                           encoding='ISO-8859-1')

    newsgroups_test = datasets.load_files(test_file,
                                          encoding='ISO-8859-1')

    remove_headers(newsgroups_test.data)
    remove_headers(newsgroups_train.data)

    stem_porter(newsgroups_test.data)
    stem_porter(newsgroups_train.data)



    #Chose f_classif over regression as it works better for this classfication problem
    anova_filter = SelectPercentile(f_classif)

    #Frequency vectorizer
    #Logarithmic value
    #Threshold to consider for df values

    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, smooth_idf=True, sublinear_tf=True, max_df=0.97,min_df=0.0000001,
                                  binary=True)
    bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words="english", lowercase=True, analyzer='word',
                                        smooth_idf=False, sublinear_tf=True, max_df=0.95, min_df=2)


    vectors = vectorizer.fit_transform(newsgroups_train.data)
    vectors_test = vectorizer.transform(newsgroups_test.data)

    pprint(vectors.shape)

    vectors_bigram = bigram_vectorizer.fit_transform(newsgroups_train.data)
    vectors_test_bigram = bigram_vectorizer.transform(newsgroups_test.data)



    svm_classifier= svm.LinearSVC(dual=False,C=1.22,tol=1e-3)

    anova_svm = Pipeline([('anova', anova_filter), ('svc', svm_classifier)])
    anova_svm.set_params(anova__percentile=90, svc__C=1.22,svc__tol=1e-5,svc__dual=False,svc__intercept_scaling=2.3).fit(vectors, newsgroups_train.target)
    #Persisting model and vectorizer

    joblib.dump(anova_svm, 'my_model.pkl', compress=9)
    joblib.dump(vectorizer,'my_vect.pkl',compress=9)


    pred_svm = anova_svm.predict(vectors_test)



    precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
    recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
    f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

    pprint("======Unigram SVM actual model======")
    pprint(precision_score)
    pprint(recall_score)
    pprint(f1_score)


    #Bigram Test with same vectorizer

    anova_svm.set_params(anova__percentile=91, svc__C=1.18,svc__tol=1e-5,svc__dual=False,svc__intercept_scaling=2.3).fit(vectors_bigram, newsgroups_train.target)
    pred_svm = anova_svm.predict(vectors_test_bigram)
    precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
    recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
    f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

    pprint("======Bigram SVM self reference======")
    pprint(precision_score)
    pprint(recall_score)
    pprint(f1_score)