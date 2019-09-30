# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 00:37:32 2019

@author: superray
"""


import statistics 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

wnl = nltk.wordnet.WordNetLemmatizer()
def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'
    
def lemmatize_sent(list_word):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(list_word)]


from nltk import word_tokenize,pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
analyzer = CountVectorizer().build_analyzer()


def stem_rmv_digit(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if not word.isdigit())

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


from sklearn.decomposition import TruncatedSVD





def plot_roc(fpr, tpr):
    fig, ax = plt.subplots()

    roc_auc = auc(fpr,tpr)

    ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)

    ax.grid(color='0.7', linestyle='--', linewidth=1)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)

    ax.legend(loc="lower right")

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)

def fit_predict_and_plot_roc(pipe, train_data, train_label, test_data, test_label):
    pipe.fit(train_data, train_label)
    predict2 = pipe.predict(test_data)
    print("%-12s %f" % ('Accuracy:', accuracy_score(test_label, predict2)))
    print("%-12s %f" % ('Precision:', precision_score(test_label, predict2, average='macro')))
    print("%-12s %f" % ('Recall:', recall_score(test_label, predict2, average='macro')))
    print("%-12s %f" % ('F1_score:', f1_score(test_label, predict2)))
    print("Confusion Matrix: \n{0}".format(confusion_matrix(test_label, predict2)))

    if hasattr(pipe, 'decision_function'):
        prob_score = pipe.decision_function(test_data)
        fpr, tpr, _ = roc_curve(test_label, prob_score)
    else:
        prob_score = pipe.predict_proba(test_data)
        fpr, tpr, _ = roc_curve(test_label, prob_score[:,1])

    plot_roc(fpr, tpr)

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))


vfunc = np.vectorize(lambda t : int(t / 4))
y_train = vfunc(newsgroups_train.target)
y_test = vfunc(newsgroups_test.target)
min_df = 3

#Soft and Hard SVM
pp1 = Pipeline([
    ('vect', CountVectorizer(min_df=min_df,analyzer=stem_rmv_digit, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50, random_state=0)),
    ('clf', SVC(kernel='linear', C=1000)),
])
pp2 = Pipeline([
    ('vect', CountVectorizer(min_df=min_df,analyzer=stem_rmv_digit, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50, random_state=0)),
    ('clf', SVC(kernel='linear', C=0.0001)),
])

fit_predict_and_plot_roc(pp1, newsgroups_train.data, y_train, newsgroups_test.data, y_test)
fit_predict_and_plot_roc(pp2, newsgroups_train.data, y_train, newsgroups_test.data, y_test)


# 5-fold cross validation
scores = []
for i in range(7):
    pipe = Pipeline([
        ('vect', CountVectorizer(min_df=min_df,analyzer=stem_rmv_digit, stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('reduce_dim', TruncatedSVD(n_components=50, random_state=0)),
        ('clf', SVC(kernel='linear', C=0.001*(10**i))),
    ])
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42, train_size = None)
    scores.append(statistics.mean(cross_val_score(pipe, newsgroups_train.data, y_train, cv=cv, scoring='f1_macro')))
    
optimal_C = 0.001* (10**scores.index(max(scores)))

print(scores)
print("the optimal C is %f" % optimal_C)

pp = Pipeline([
    ('vect', CountVectorizer(min_df=min_df,analyzer=stem_rmv_digit, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50, random_state=0)),
    ('clf', SVC(kernel='linear', C=optimal_C)),
])
        
fit_predict_and_plot_roc(pp, newsgroups_train.data, y_train, newsgroups_test.data, y_test)
