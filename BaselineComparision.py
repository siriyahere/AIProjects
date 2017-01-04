from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
import matplotlib.pyplot as plt



def remove_headers(data):
    for i, text in enumerate(data):
        _before, _blankline, after = text.partition('\n\n')
        data[i]=after

newsgroups_train = datasets.load_files("dataset/Training",
                                       encoding='ISO-8859-1')

newsgroups_test = datasets.load_files("dataset/Test",
                                      encoding='ISO-8859-1')

remove_headers(newsgroups_test.data)
remove_headers(newsgroups_train.data)

vectorizer = TfidfVectorizer()
bigram_vectorizer=TfidfVectorizer(ngram_range=(2, 2))



vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)

vectors_bigram = bigram_vectorizer.fit_transform(newsgroups_train.data)
vectors_test_bigram = bigram_vectorizer.transform(newsgroups_test.data)

clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
pred = clf.predict(vectors_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred, average='macro')

pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


linear = linear_model.LogisticRegression(C=1e+6, penalty='l2', tol=1e-6)
linear.fit(vectors, newsgroups_train.target)
pred_linear = linear.predict(vectors_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_linear, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_linear, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_linear, average='macro')

pprint("======Logistic Regression======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)

svm_classifier= svm.LinearSVC()
svm_classifier.fit(vectors, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======SVM======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


randomForest_classifier=ensemble.RandomForestClassifier()
randomForest_classifier.fit(vectors, newsgroups_train.target)
pred_randomForest = randomForest_classifier.predict(vectors_test)
precision_score = metrics.precision_score(newsgroups_test.target, pred_randomForest, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_randomForest, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_randomForest, average='macro')

pprint("======RandomForest======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


clf = MultinomialNB(alpha=.01)
clf.fit(vectors_bigram, newsgroups_train.target)
pred = clf.predict(vectors_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred, average='macro')

pprint("====NB bigram====")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


linear = linear_model.LogisticRegression(C=1e+5, penalty='l2', tol=2.3e-2)
linear.fit(vectors_bigram, newsgroups_train.target)
pred_linear = linear.predict(vectors_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred_linear, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_linear, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_linear, average='macro')

pprint("======Bigram Logistic Regression======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)

svm_classifier= svm.LinearSVC()
svm_classifier.fit(vectors_bigram, newsgroups_train.target)
pred_svm = svm_classifier.predict(vectors_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_svm, average='macro')

pprint("======Bigram SVM======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)


randomForest_classifier=ensemble.RandomForestClassifier()
randomForest_classifier.fit(vectors_bigram, newsgroups_train.target)
pred_randomForest = randomForest_classifier.predict(vectors_test_bigram)
precision_score = metrics.precision_score(newsgroups_test.target, pred_randomForest, average='macro')
recall_score = metrics.recall_score(newsgroups_test.target, pred_randomForest, average='macro')
f1_score= metrics.f1_score(newsgroups_test.target, pred_randomForest, average='macro')

pprint("======Bigram RandomForest======")
pprint(precision_score)
pprint(recall_score)
pprint(f1_score)

train_sizes=range(100,2170,100)
NB_f1_scores=[]
RF_f1_scores =[]
LR_f1_scores =[]
SVM_f1_scores =[]


clf = MultinomialNB(alpha=.01)
for i in train_sizes:

    vectors = vectorizer.fit_transform(newsgroups_train.data[:i])
    vectors_test = vectorizer.transform(newsgroups_test.data)

    clf.fit(vectors, newsgroups_train.target[:i])
    pred = clf.predict(vectors_test)
    precision_score = metrics.precision_score(newsgroups_test.target, pred, average='macro')
    recall_score = metrics.recall_score(newsgroups_test.target, pred, average='macro')
    NB_f1_scores.append(metrics.f1_score(newsgroups_test.target, pred, average='macro'))

linear = linear_model.LogisticRegression()
for i in train_sizes:

    vectors = vectorizer.fit_transform(newsgroups_train.data[:i])
    vectors_test = vectorizer.transform(newsgroups_test.data)

    linear.fit(vectors, newsgroups_train.target[:i])
    pred = linear.predict(vectors_test)
    precision_score = metrics.precision_score(newsgroups_test.target, pred, average='macro')
    recall_score = metrics.recall_score(newsgroups_test.target, pred, average='macro')
    LR_f1_scores.append(metrics.f1_score(newsgroups_test.target, pred, average='macro'))

svm_classifier = svm.LinearSVC()
for i in train_sizes:

    vectors = vectorizer.fit_transform(newsgroups_train.data[:i])
    vectors_test = vectorizer.transform(newsgroups_test.data)

    svm_classifier.fit(vectors, newsgroups_train.target[:i])
    pred_svm = svm_classifier.predict(vectors_test)

    precision_score = metrics.precision_score(newsgroups_test.target, pred_svm, average='macro')
    recall_score = metrics.recall_score(newsgroups_test.target, pred_svm, average='macro')
    SVM_f1_scores.append(metrics.f1_score(newsgroups_test.target, pred_svm, average='macro'))


randomForest_classifier=ensemble.RandomForestClassifier()
for i in train_sizes:
    vectors = vectorizer.fit_transform(newsgroups_train.data[:i])
    vectors_test = vectorizer.transform(newsgroups_test.data)
    randomForest_classifier.fit(vectors, newsgroups_train.target[:i])
    pred_randomForest = randomForest_classifier.predict(vectors_test)
    precision_score = metrics.precision_score(newsgroups_test.target, pred_randomForest, average='macro')
    recall_score = metrics.recall_score(newsgroups_test.target, pred_randomForest, average='macro')
    RF_f1_scores.append(metrics.f1_score(newsgroups_test.target, pred_randomForest, average='macro'))


title = "LEARNING CURVES"
plt.figure()
plt.title(title)
plt.xlabel("TEST DATASET SIZE")
plt.ylabel("F1 SCORE")
ylim=(0.3,1.1)
plt.ylim(*ylim)
plt.grid()

plt.plot(train_sizes, NB_f1_scores, 'o-', color="r",label="NAIVE BAYES")
plt.plot(train_sizes, LR_f1_scores, 'o-', color="b",label="LOGISTIC_REGRESSION")

plt.plot(train_sizes, SVM_f1_scores, 'o-', color="g",
             label="SVM")

plt.plot(train_sizes, RF_f1_scores, 'o-', color="y",
             label="Random Forest")
plt.legend(loc="best")
plt.show()