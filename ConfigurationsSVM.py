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
from nltk.stem.porter import *


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


newsgroups_train = datasets.load_files("dataset/Training",
                                       encoding='ISO-8859-1')

newsgroups_test = datasets.load_files("dataset/Test",
                                      encoding='ISO-8859-1')

remove_headers(newsgroups_test.data)
remove_headers(newsgroups_train.data)



count_vectorizer = CountVectorizer(stop_words="english",lowercase=True)
count_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words="english",lowercase=True,analyzer='word')




tfid_vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
tfid_bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words="english", lowercase=True)


pprint("##################Without Porter Stemming######################")

vectors_count = count_vectorizer.fit_transform(newsgroups_train.data)
vectors_count_test = count_vectorizer.transform(newsgroups_test.data)

vectors_count_bigram = count_bigram_vectorizer.fit_transform(newsgroups_train.data)
vectors_count_test_bigram = count_bigram_vectorizer.transform(newsgroups_test.data)

svm_classifier= svm.LinearSVC(penalty='l2')
#svm_classifier= svm.SVC()
svm_classifier.fit(vectors_count, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_count_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Count Vectorizer======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)





svm_classifier= svm.LinearSVC(penalty='l2',C=1)
svm_classifier.fit(vectors_count_bigram, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_count_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Bigram Count Vectorizer======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)






vectors_count = count_vectorizer.fit_transform(newsgroups_train.data)
vectors_count_test = count_vectorizer.transform(newsgroups_test.data)

vectors_count_bigram = count_bigram_vectorizer.fit_transform(newsgroups_train.data)
vectors_count_test_bigram = count_bigram_vectorizer.transform(newsgroups_test.data)



vectors = tfid_vectorizer.fit_transform(newsgroups_train.data)
vectors_test = tfid_vectorizer.transform(newsgroups_test.data)

vectors_bigram = tfid_bigram_vectorizer.fit_transform(newsgroups_train.data)
vectors_test_bigram = tfid_bigram_vectorizer.transform(newsgroups_test.data)

svm_classifier= svm.LinearSVC(penalty='l2',C=1)
svm_classifier.fit(vectors, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Tfid Vectorizer======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


svm_classifier= svm.LinearSVC(penalty='l2')
svm_classifier.fit(vectors_bigram, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Bigram TFID SVM======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)



svm_classifier= svm.LinearSVC(penalty='l2',C=1.22)
#svm_classifier= svm.SVC()
svm_classifier.fit(vectors, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Tfid Vectorizer Penalty l2 C =1.22======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


svm_classifier= svm.LinearSVC(penalty='l2',C=1.22)
svm_classifier.fit(vectors_bigram, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Bigram TFID SVM Penalty l2 C=1.22======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)

svm_classifier= svm.LinearSVC(penalty='l1',C=1.0,dual=False)
#svm_classifier= svm.SVC()
svm_classifier.fit(vectors, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Tfid Vectorizer Penalty L1 dual false======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


svm_classifier= svm.LinearSVC(penalty='l1',C=1.00,dual=False)
svm_classifier.fit(vectors_bigram, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Bigram TFID SVM Penalty L1 dual false======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)

pprint("###########With Porter Stemming#####################")

stem_porter(newsgroups_test.data)
stem_porter(newsgroups_train.data)


count_vectorizer = CountVectorizer(stop_words="english",lowercase=True)
count_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words="english",lowercase=True,analyzer='word')




tfid_vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
tfid_bigram_vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words="english", lowercase=True)




vectors_count = count_vectorizer.fit_transform(newsgroups_train.data)
vectors_count_test = count_vectorizer.transform(newsgroups_test.data)

vectors_count_bigram = count_bigram_vectorizer.fit_transform(newsgroups_train.data)
vectors_count_test_bigram = count_bigram_vectorizer.transform(newsgroups_test.data)

svm_classifier= svm.LinearSVC(penalty='l2')
#svm_classifier= svm.SVC()
svm_classifier.fit(vectors_count, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_count_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Count Vectorizer======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)





svm_classifier= svm.LinearSVC(penalty='l2',C=1)
svm_classifier.fit(vectors_count_bigram, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_count_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Bigram Count Vectorizer======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


vectors_count = count_vectorizer.fit_transform(newsgroups_train.data)
vectors_count_test = count_vectorizer.transform(newsgroups_test.data)

vectors_count_bigram = count_bigram_vectorizer.fit_transform(newsgroups_train.data)
vectors_count_test_bigram = count_bigram_vectorizer.transform(newsgroups_test.data)



vectors = tfid_vectorizer.fit_transform(newsgroups_train.data)
vectors_test = tfid_vectorizer.transform(newsgroups_test.data)

vectors_bigram = tfid_bigram_vectorizer.fit_transform(newsgroups_train.data)
vectors_test_bigram = tfid_bigram_vectorizer.transform(newsgroups_test.data)

svm_classifier= svm.LinearSVC(penalty='l2',C=1)
svm_classifier.fit(vectors, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Tfid Vectorizer======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


svm_classifier= svm.LinearSVC(penalty='l2')
svm_classifier.fit(vectors_bigram, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Bigram TFID SVM======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)



svm_classifier= svm.LinearSVC(penalty='l2',C=1.22)
#svm_classifier= svm.SVC()
svm_classifier.fit(vectors, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Tfid Vectorizer Penalty l2 C =1.22======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


svm_classifier= svm.LinearSVC(penalty='l2',C=1.22)
svm_classifier.fit(vectors_bigram, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Bigram TFID SVM Penalty l2 C=1.22======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)

svm_classifier= svm.LinearSVC(penalty='l1',C=1.0,dual=False)
#svm_classifier= svm.SVC()
svm_classifier.fit(vectors, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Tfid Vectorizer Penalty L1 dual false======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


svm_classifier= svm.LinearSVC(penalty='l1',C=1.00,dual=False)
svm_classifier.fit(vectors_bigram, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Bigram TFID SVM Penalty L1 dual false======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)

