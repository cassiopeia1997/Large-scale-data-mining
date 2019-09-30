#!/usr/bin/env python
# coding: utf-8

# # Multiclass Classification:

# # Naïve Bayes

# In[85]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

categories_multi= ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 
              'misc.forsale', 'soc.religion.christian']
train_dataset = fetch_20newsgroups(subset = 'train', categories = categories_multi, shuffle = True, random_state = 42)
test_dataset = fetch_20newsgroups(subset = 'test', categories = categories_multi, shuffle = True, random_state = 42)

class SparseToDenseArray(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, *_):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X

    def fit(self, *_):
        return self
    
    
pipeline_Bayes = Pipeline([
    ('vect', CountVectorizer(min_df=3, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50, random_state=0)),
    ('toarr', SparseToDenseArray()),
    ('clf', MultinomialNB()),
])


pipeline_Bayes.set_params(reduce_dim=None)
pipeline_Bayes.fit(train_dataset.data, train_dataset.target)
predict = pipeline_Bayes.predict(test_dataset.data)
print("Accuracy:{}".format(accuracy_score(test_dataset.target, predict)))
print("Recall:{}".format(recall_score(test_dataset.target, predict,average='macro')))
print("Precision:{}".format(precision_score(test_dataset.target, predict,average='macro')))
print("F-1:{}".format(f1_score(test_dataset.target, predict,average='macro')))
print("Confusion Matrix: \n{0}".format(confusion_matrix(test_dataset.target, predict)))


# # MultiClass SVM - One VS One

# In[86]:



pipeline_svm1 = Pipeline([
    ('vect', CountVectorizer(min_df=1, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50, random_state=0)),
    ('toarr', SparseToDenseArray()),
    ('clf', SVC(kernel='linear', C=10,decision_function_shape='ovo')),
])

pipeline_svm1.fit(train_dataset.data, train_dataset.target)
predict = pipeline_svm1.predict(test_dataset.data)
print("Accuracy:{}".format(accuracy_score(test_dataset.target, predict)))
print("Recall:{}".format(recall_score(test_dataset.target, predict,average='macro')))
print("Precision:{}".format(precision_score(test_dataset.target, predict,average='macro')))
print("F-1:{}".format(f1_score(test_dataset.target, predict,average='macro')))
print("Confusion Matrix: \n{0}".format(confusion_matrix(test_dataset.target, predict)))


# # MultiClass SVM - One to Rest

# In[83]:




pipeline_svm2 = Pipeline([
    ('vect', CountVectorizer(min_df=1, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50, random_state=0)),
    ('toarr', SparseToDenseArray()),
    ('clf', LinearSVC(C=10)),
])

pipeline_svm2.fit(train_dataset.data, train_dataset.target)
predict2 = pipeline_svm2.predict(test_dataset.data)
print("Accuracy:{}".format(accuracy_score(test_dataset.target, predict2)))
print("Recall:{}".format(recall_score(test_dataset.target, predict2,average='macro')))
print("Precision:{}".format(precision_score(test_dataset.target, predict2,average='macro')))
print("F-1:{}".format(f1_score(test_dataset.target, predict2,average='macro')))
print("Confusion Matrix: \n{0}".format(confusion_matrix(test_dataset.target, predict2)))

