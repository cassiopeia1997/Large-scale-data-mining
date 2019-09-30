#!/usr/bin/env python
# coding: utf-8

# In[6]:


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

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np 
np.random.seed(42) 
import random 
random.seed(42)

categories = ['comp.graphics',
'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware','rec.autos',
'rec.motorcycles','rec.sport.baseball',
'rec.sport.hockey']

train_dataset = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42)
test_dataset = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = 42)

class SparseToDenseArray(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, *_):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X

    def fit(self, *_):
        return self

target2_names=['Computer technology','Recreational activity']
target2=np.zeros([4732,1])
for i in range(0,len(train_dataset.target)):
    if train_dataset.target[i]<=3:
        target2[i]=0
    else:
        target2[i]=1
target2=np.int_(target2).ravel()
target2.shape

target2_test=np.zeros([3150,1])
for i in range(0,len(test_dataset.target)):
    if test_dataset.target[i]<=3:
        target2_test[i]=0
    else:
        target2_test[i]=1
target2_test=np.int_(target2_test).ravel()
target2_test.shape
    
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
    # pipeline1.predict(twenty_test.data)

    if hasattr(pipe, 'decision_function'):
        prob_score = pipe.decision_function(test_data)
        fpr, tpr, _ = roc_curve(test_label, prob_score)
    else:
        prob_score = pipe.predict_proba(test_data)
        fpr, tpr, _ = roc_curve(test_label, prob_score[:,1])

    plot_roc(fpr, tpr)
    
pipeline_Bayes = Pipeline([
    ('vect', CountVectorizer(min_df=3, stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', TruncatedSVD(n_components=50, random_state=0)),
    ('toarr', SparseToDenseArray()),
    ('clf', GaussianNB()),
])


pipeline_Bayes.fit(train_dataset.data, target2)
predict = pipeline_Bayes.predict(test_dataset.data)
print("Accuracy:{}".format(accuracy_score(target2_test, predict)))
print("Recall:{}".format(recall_score(target2_test, predict)))
print("Precision:{}".format(precision_score(target2_test, predict)))
print("F-1:{}".format(f1_score(target2_test, predict)))
print("Confusion Matrix: \n{0}".format(confusion_matrix(target2_test, predict)))
fit_predict_and_plot_roc(pipeline_Bayes, train_dataset.data, target2, test_dataset.data,target2_test)


# In[ ]:




