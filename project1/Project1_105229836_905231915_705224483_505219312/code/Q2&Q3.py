import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize,pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize,pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

np.random.seed(42)
random.seed(42)

categories = ['comp.graphics',
'comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware','rec.autos',
'rec.motorcycles','rec.sport.baseball',
'rec.sport.hockey']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


wnl=WordNetLemmatizer()
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def tokenize_and_lem(sentence):
    tokens = word_tokenize(sentence)
    tokens=[token for token in tokens if not token.isdigit()]
    tagged_sent=pos_tag(tokens)
    lemmas_sent=[]
    for tag in tagged_sent:
        wordnet_pos=get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0],pos=wordnet_pos))
    return lemmas_sent

wnl = nltk.wordnet.WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'

def lemmatize_sent_demo(text):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag))
            for word, tag in pos_tag(nltk.word_tokenize(text))]
def lemmatize_sent(list_word):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag))
            for word, tag in pos_tag(list_word)]


analyzer = CountVectorizer().build_analyzer()


def stem_rmv_digit(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if not word.isdigit())

min_df = 3
countVectorizer = CountVectorizer(min_df=min_df,analyzer=stem_rmv_digit, stop_words='english')
countVectorTrain = countVectorizer.fit_transform(newsgroups_train.data)
print("training dataset: TF-ICF")
print(countVectorTrain.shape)

countVectorsTest = countVectorizer.transform(newsgroups_test.data)
#print('Test dataset: When min_df=%d, we get %d documents with %d terms.' % (min_df, countVectorsTest.shape[0], countVectorsTest.shape[1]))
print("test dataset:TF-ICF")
print(countVectorsTest.shape)
print("")
print("TF-IDF")
tfidf_transformer = TfidfTransformer()
tficfVectorsTrain = tfidf_transformer.fit_transform(countVectorTrain)
tficfVectorsTest = tfidf_transformer.transform(countVectorsTest)
print("training dataset:")
print(tficfVectorsTrain.shape)
print("test dataset")
print(tficfVectorsTest.shape)
print("")
print("Reduce Dimension:LSI")
svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
lsiVectorsTrain = svd.fit_transform(tficfVectorsTrain)
print(lsiVectorsTrain)
print(lsiVectorsTrain.shape)
b=svd.inverse_transform(lsiVectorsTrain)
print("error:")
print(np.linalg.norm(tficfVectorsTrain-b,ord = 'fro'))
print("Apply on test dataset")
print(tficfVectorsTest.shape)
lsiVectorsTest = svd.transform(tficfVectorsTest)
print(lsiVectorsTest)
print(lsiVectorsTest.shape)
print("")
print("Reduce Dimension:NMF")
nmf = NMF(n_components=50, init='random', random_state=0)
nmfVectorsTrain = nmf.fit_transform(tficfVectorsTrain)
print(nmfVectorsTrain)
print(nmfVectorsTrain.shape)
print("error:")
b=nmf.inverse_transform(nmfVectorsTrain)
print(b.shape)
print(np.linalg.norm(tficfVectorsTrain-b,ord = 'fro'))
print("Apply on test dataset")
nmfVectorsTest = nmf.transform(tficfVectorsTest)
print(nmfVectorsTest)
print(nmfVectorsTest.shape)