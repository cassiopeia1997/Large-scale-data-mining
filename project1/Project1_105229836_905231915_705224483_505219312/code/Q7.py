#!/usr/bin/env python
# coding: utf-8

# # pipeline and grid search

# In[9]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, NMF
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

twenty_train = fetch_20newsgroups(subset='train', 
                                  categories=categories, 
                                  shuffle=True,
                                  random_state=42, 
                                  remove = ['headers','footers']
                                 )
vfunc = np.vectorize(lambda t : t / 4)
y_train = vfunc(twenty_train.target)

# used to cache results
#from tempfile import mkdtemp
#from shutil import rmtree
#from sklearn.externals.joblib import Memory
# print(__doc__)
#cachedir = mkdtemp()
#memory = Memory(cachedir=cachedir, verbose=10)

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(random_state=42)),
    ('clf', GaussianNB()),
],
#memory=memory
)
C_OPTIONS = [0.1, 1, 10]
param_grid = [
    {
        #'loaddata__remove':[('headers','footers'), None],
        'vect__min_df':[3, 5],
        'reduce_dim': [TruncatedSVD(n_components=50), NMF(n_components=50)],
        'clf': [LinearSVC()],
        'clf__C': C_OPTIONS
    },
    {
        #'loaddata__remove':[('headers','footers'), None],
        'vect__min_df':[3, 5],
        'reduce_dim': [TruncatedSVD(n_components=50), NMF(n_components=50)],
        'clf': [GaussianNB()],
    },
    {
        #'loaddata__remove':[('headers','footers'), None],
        'vect__min_df':[3, 5],
        'reduce_dim': [TruncatedSVD(n_components=50), NMF(n_components=50)],
        'clf': [LogisticRegression()],
        'clf__C': C_OPTIONS,
        'clf__penalty': ['l1', 'l2']
    },
]
# reducer_labels = ['LinearSVC', 'NMF', 'KBest(chi2)']

grid = GridSearchCV(pipeline, cv=3, n_jobs=1, param_grid=param_grid, scoring='accuracy')
grid.fit(twenty_train.data, y_train.astype('int'))
print("\nBest Parameters: \n{0}\n".format(grid.best_params_))
print("\nBest Accuracy: \n{0}\n".format(grid.best_score_))
#rmtree(cachedir)


# In[10]:


import pandas as pd

pd.DataFrame(grid.cv_results_)


# In[11]:





