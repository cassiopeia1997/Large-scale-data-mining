# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 02:32:55 2019

@author: superray
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk import word_tokenize,pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
wnl = nltk.wordnet.WordNetLemmatizer()
analyzer = CountVectorizer().build_analyzer()

def stem_rmv_digit(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if not word.isdigit())

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

'''
actual code below
'''
min_df = 3
pp = Pipeline([
    ('vect', CountVectorizer(min_df=min_df,analyzer=stem_rmv_digit, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50, random_state=0)),
    ('clf', LogisticRegression(penalty = 'l2',C = 1000000.0)),
])
        
# merge data into two groups
vfunc = np.vectorize(lambda t : int(t / 4))
y_train = vfunc(newsgroups_train.target)
y_test = vfunc(newsgroups_test.target)


fit_predict_and_plot_roc(pp, newsgroups_train.data, y_train, newsgroups_test.data, y_test)

# grid search
import numpy as np
np.random.seed(42)
import random
random.seed(42)
from sklearn.datasets import fetch_20newsgroups
# Refer to the offcial document of scikit-learn for detailed usages:
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
# The
newsgroups_train = fetch_20newsgroups(subset='train', # choose which subset of the dataset to use; can be 'train', 'test', 'all'
                                  categories=categories, # choose the categories to load; if is `None`, load all categories
                                  shuffle=True,
                                  random_state=42, # set the seed of random number generator when shuffling to make the outcome repeatable across different runs
#                                   remove=['headers'],
                                 )
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
import string
nltk.download('punkt')

stemmer = PorterStemmer()
# stemmer = SnowballStemmer('english')
def tokenize_and_stem(text):
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [token.strip(string.punctuation) for token in tokens if token.isalnum()]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens
print(newsgroups_train.data[1])
print(tokenize_and_stem(newsgroups_train.data[1]))
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

min_df = 3
tfidfVectorizer = TfidfVectorizer(min_df=min_df, stop_words='english', tokenizer=tokenize_and_stem)
tfidfVectors = tfidfVectorizer.fit_transform(newsgroups_train.data)
print('When min_df=%d, we get %d documents with %d terms.' % (min_df, tfidfVectors.shape[0], tfidfVectors.shape[1]))
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
lsiVectors = svd.fit_transform(tfidfVectors)

print('When min_df=%d, we get %d documents, each with a %d-dimensional feature vector.' % (min_df, lsiVectors.shape[0], lsiVectors.shape[1]))
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

# preprocessing pipeline
pp = Pipeline([('vect', tfidfVectorizer), ('dim_red', svd), ('norm', Normalizer(copy=False))])

x_train = pp.fit_transform(newsgroups_train.data)
x_test = pp.transform(newsgroups_test.data)

vfunc = np.vectorize(lambda t : t / 4)
y_train = vfunc(newsgroups_train.target)
y_test = vfunc(newsgroups_test.target)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import math
parameters = {'penalty': ['l2'],
              'C': [math.pow(10, k) for k in range(-3,4)]}

gs_clf = GridSearchCV(LogisticRegression(), parameters, n_jobs=-1, cv=5)
gs_clf.fit(x_train, y_train.astype('int'))
score = gs_clf.predict_proba(x_test)[:,1]
predicted = gs_clf.predict(x_test)

values = [list(gs_clf.cv_results_['param_penalty']),
          list(gs_clf.cv_results_['param_C']),
          ["%.6f" % x for x in gs_clf.cv_results_['mean_test_score']],
          list(gs_clf.cv_results_['rank_test_score'])]

#draw_table(values, title='Logistic Regression Grid Search Scores')

print("\nBest Parameters: \n{0}\n".format(gs_clf.best_params_))
print("%-12s %f" % ('Accuracy:', metrics.accuracy_score(y_test.astype('int'), predicted)))
print("%-12s %f" % ('Precision:', metrics.precision_score(y_test.astype('int'), predicted, average='macro')))
print("%-12s %f" % ('Recall:', metrics.recall_score(y_test.astype('int'), predicted, average='macro')))
print("%-12s %f" % ('F1 Score:', metrics.f1_score(y_test.astype('int'), predicted, average='macro')))
print("Confusion Matrix: \n{0}".format(metrics.confusion_matrix(y_test.astype('int'), predicted)))

fpr_lr_best, tpr_lr_best, thresholds = metrics.roc_curve(y_test.astype('int'), score)
#roc_auc_lr_best = metrics.auc(fpr_lr_best, tpr_lr_best)
#draw_roc_curve(fpr_lr_best, tpr_lr_best, roc_auc_lr_best, 'ROC curve')
