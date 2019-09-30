import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as plt
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
print(newsgroups_train.target_names)


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

unique, counts = np.unique(newsgroups_train.target, return_counts=True)
plt.bar(unique, counts, align='center', tick_label=newsgroups_train.target_names, color='xkcd:sky blue')
plt.title('Number of Documents for Each Topic')
plt.ylabel('Number of Documents')
plt.ylim([400,700])
plt.xticks(rotation='vertical')
plt.yticks(np.arange(400, 700, 50))
plt.show()



