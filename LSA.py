from __future__ import print_function

import sklearn.cross_validation
import sklearn.datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction import text

from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import pylab as pl


from scipy.sparse.linalg import svds
from scipy.stats.stats import pearsonr

import sklearn as sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.utils.extmath import randomized_svd,svd_flip
import matplotlib.pyplot as plt

from sklearn import metrics

import logging
from optparse import OptionParser
import sys
from time import time
from sklearn.ensemble import RandomForestClassifier

import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

import warnings


old_stdout = sys.stdout
log_file = open("message.log","w")
sys.stdout = log_file

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Display progress logs on stdout

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",
              action="store_true", default=False,
              help="Use a hashing feature vectorizer")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False,
              help="Print progress reports inside k-means algorithm.")

print(__doc__)
op.print_help()


(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)


###############################################################################
# Load some categories from the training set

categories = [
    'comp.sys.mac.hardware',
    'rec.sport.baseball',
    'sci.med',
    'talk.politics.guns',
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
    'comp.os.ms-windows.misc',
    'rec.autos',
    'sci.crypt',
    'misc.forsale',
    'comp.sys.ibm.pc.hardware',
    'rec.motorcycles',
    'sci.electronics',
    'talk.politics.misc',
    'comp.windows.x',
    'rec.sport.hockey',
    'talk.politics.mideast',
    'soc.religion.christian'
]

'''
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]



categories = [
    'comp.os.ms-windows.misc',
    'rec.autos',
    'sci.crypt',
    'misc.forsale'
]


categories = [
    'comp.sys.ibm.pc.hardware',
    'rec.motorcycles',
    'sci.electronics',
    'talk.politics.misc'
]




categories = [
    'comp.sys.mac.hardware',
    'rec.sport.baseball',
    'sci.med',
    'talk.politics.guns'
]



categories = [
    'comp.windows.x',
    'rec.sport.hockey',
    'talk.politics.mideast',
    'soc.religion.christian'
]
'''


# Uncomment the following to do the analysis on all the categories
#categories = None

def print_metrics(scores):
    """
    Compute and print evaluation metrics.
    Accuracy, Precision, Recall and F1.
    """
    print

    print("--------- Final results for Cross Validation ---------")

    print("Avg Accuracy", (np.sum(scores["accuracy"]) / 2))

    print ("Avg Precision", (np.sum(scores["precision"]) / 2))

    print("Avg Recall", (np.sum(scores["recall"]) / 2))

    print("Avg F1", (np.sum(scores["f1"]) / 2))
    print

def plot_confusion_matrix(cm):
    # Print confusion matrix from last result
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()


###########################Performing the classification task#########################################
def classificjobfortestdata(X,y):
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    testdataset = fetch_20newsgroups(subset='test', categories=categories,
                                 shuffle=True, random_state=42)

    vectorizer2 = TfidfVectorizer(stop_words=stop, min_df=0.018, max_df=0.50, lowercase=1, max_features=5000,
                                  tokenizer=StemTokenizer())
    X2 = vectorizer2.fit_transform(testdataset.data)

    train_text = X
    train_target = y

    test_text = X2
    test_target = testdataset.target

    text_clf = SVC(C=1.9, kernel='linear')
    _ = text_clf.fit(train_text, train_target)

    # predict
    predicted = text_clf.predict(test_text)

    # compute metrics
    accuracy = metrics.accuracy_score(test_target, predicted)
    precision = metrics.precision_score(test_target, predicted, average='macro')
    recall = metrics.recall_score(test_target, predicted, average='macro')
    f1 = metrics.f1_score(test_target, predicted, average='macro')

    scores["accuracy"].append(accuracy)
    scores["precision"].append(precision)
    scores["recall"].append(recall)
    scores["f1"].append(f1)

    print("**** Classification report ****")

    print(metrics.classification_report(test_target, predicted,
                                        target_names=dataset.target_names))

    # Compute confusion matrix

    print("**** Confusion matrix ****")
    cm = confusion_matrix(test_target, predicted)
    print(cm)
    print_metrics(scores)
    return


def classificjob(X,y) :
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    print("the data is being printed ")
    ss = sklearn.cross_validation.ShuffleSplit((X.shape[0]),
                                               n_iter=2, test_size=0.2,
                                               random_state=1234)
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}


    print("Number of documents for training", (1 - 0.2) * len(dataset.data))

    print("Number of documents for testing", 0.2 * len(dataset.data))

    fold = 0
    for train_index, test_index in ss:
        fold += 1

        print("Fold", fold, "--------------------------------")

        # get the text and label for the training data
        train_text = X[[x for x in train_index] ,:]
        train_target = y[[x for x in train_index],:]

        test_text = X[[x for x in test_index], :]
        test_target = y[[x for x in test_index], :]


        #train_text = [[X[x]] for x in train_index]
        #train_target = [y[x] for x in train_index]

        # label and text for the test data
        #test_text = [[X[x]] for x in test_index]
        #test_target = [y[x] for x in test_index]



        '''


        vectorizer = CountVectorizer(max_df=1.0,

                                 stop_words='english',
                                 lowercase=True,
                                 strip_accents="unicode",
                                 ngram_range=(1, 5))

        vect = CountVectorizer(ngram_range=(1, 5), max_df=1.0)
        '''
        '''
        text_clf = Pipeline([('vect', vectorizer),
                         ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
                         ('clf', SVC(C=1.9, kernel='linear')), ])




        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 5), max_df=1.0)),
                             ('tfidf', TfidfTransformer(use_idf=True)),
                             ('clf', MultinomialNB()), ])
        '''
        # train



        text_clf = SVC(C=1.9, kernel='linear')
        _ = text_clf.fit(train_text, train_target)

        # predict
        predicted = text_clf.predict(test_text)

        # compute metrics
        accuracy = metrics.accuracy_score(test_target, predicted)
        precision = metrics.precision_score(test_target, predicted, average='macro')
        recall = metrics.recall_score(test_target, predicted, average='macro')
        f1 = metrics.f1_score(test_target, predicted, average='macro')

        scores["accuracy"].append(accuracy)
        scores["precision"].append(precision)
        scores["recall"].append(recall)
        scores["f1"].append(f1)

        print("**** Classification report ****")

        print(metrics.classification_report(test_target, predicted,
                                        target_names=dataset.target_names))

        # Compute confusion matrix

        print("**** Confusion matrix ****")
        cm = confusion_matrix(test_target, predicted)
        print(cm)

    print_metrics(scores)


class StemTokenizer(object):
    def __init__(self):
        self.stm = SnowballStemmer("english")

    def __call__(self, doc):
        list1 = [self.stm.stem(t) for t in word_tokenize(doc)]
        list2 = []
        for a in list1:
            fl = 0
            for i in range(len(a)):
                if (not (((a[i] >= 'a') and (a[i] <= 'z')) or ((a[i] >= 'A') and (a[i] <= 'Z')))):
                    fl = 1
                    break
            if fl == 0:
                list2.append(a)
        # return [self.stm.stem(t) for t in word_tokenize(doc)]

        return list2


stp = {"abc","hrs","af","are","ar","aa", "aaa","ab","abl","aap","adb","zz", "think","does","just","like","said","did","the","they","our","not","a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"}
stop = text.ENGLISH_STOP_WORDS.union(stp)

print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

#print (dataset.data[0])

print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))

labels = dataset.target
true_k = np.unique(labels).shape[0] # because it (shape) is a tuple

#print (dataset.data[100])

print("Extracting features from the training dataset using a sparse vectorizer")

t0 = time()
'''
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       non_negative=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(stop_words=stop, min_df = 0.01, max_df=0.50,lowercase=1,max_features=opts.n_features,tokenizer=StemTokenizer())
'''

#vectorizer = TfidfVectorizer(stop_words=stop, min_df = 0.018,max_df=0.50,lowercase=1,max_features=opts.n_features,tokenizer=StemTokenizer())

vectorizer = TfidfVectorizer(stop_words=stop,min_df=0.001,max_df=0.15,lowercase=1,tokenizer=StemTokenizer())
X = vectorizer.fit_transform(dataset.data)
rawdata = X

li = vectorizer.get_feature_names()
print(li)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

res = []
opts.n_components = min(X.shape[0],X.shape[1]) - 5

if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.

    """
    U, Sigma, VT = randomized_svd(X, n_components=opts.n_components,
                                 random_state=42)

    svd.fit(X)
    Vch = svd.components_
    Vch = np.asarray(Vch)
    Vcht = np.transpose(Vch)
    I = np.dot(Vch, Vcht)
    #print("old I  ")
    #print (I[0:])
    """
    U, Sigma, VT = svds(X, k=opts.n_components, tol=0)
    # svds doesn't abide by scipy.linalg.svd/randomized_svd
    # conventions, so reverse its outputs.
    Sigma = Sigma[::-1]
    U, VT = svd_flip(U[:, ::-1], VT[::-1])

    U = np.asarray(U)
    Sigma = np.asarray(Sigma)
    VT = np.asarray(VT)
    V = np.transpose(VT)

    I = np.dot(VT, V)
    print ("newly I  ")
    print(I)

    UT = np.transpose(U)

    print (VT.shape)
    print (X.shape)
    Xin = X.todense()
    Xin = np.asarray(Xin)
    print (Xin.shape)

    res = np.dot(Xin,V)

    terms = vectorizer.get_feature_names()
    termsIncomp = []

    for i, comp in enumerate(VT) :
        termsIncomp = zip(terms, comp)
        sortedterms = sorted(termsIncomp, key = lambda x: x[1], reverse=True)[:20]
        print ("concept " + str(i) + " with variance " + str(Sigma[i]))
        for term in sortedterms:
            print (str(term[0]) + "  " + str(term[1]))
        print()
        if i == 5:
           break


###############################random forest stuff #######################################33

labels = np.matrix([labels])
labels = labels.transpose()

rf = RandomForestClassifier(n_estimators=400)  # initialize
rf.fit(res, labels)

print(res.shape)
print(labels.shape)


featureImportance = rf.feature_importances_
print("********************************************************************************************")

################################### mutual information content #######################################

X, y = res, labels

print(X.shape)
sb = SelectKBest(mutual_info_classif,'all')
sb.fit(X,y)
a = sb.scores_

# if run without CV plot Confusion Matrix

#perform_classification(X)
######################################################################################################
print("Classification job for vectorized data with all features")
classificjob(rawdata,y)

mylist = list(range(10000))
print("Classification job for LSA with all features")
classificjob(X,y)

######################################################################################################

maxSigma = max(Sigma)
maxFeature = max(featureImportance)

Sigma = [x/maxSigma for x in Sigma]
featureImportance = [x/maxFeature for x in featureImportance]

x = range(len(featureImportance))

fig = plt.figure()

ax1 = fig.add_subplot(131)
ax1.set_xlabel('feature number')
ax1.set_ylabel('Random forest feature importance')
ax1.plot(x, featureImportance, 'g-')

x = range(len(Sigma))
ax2 = fig.add_subplot(132)
ax2.set_xlabel('feature number')
ax2.set_ylabel('LSA Singular Value')
ax2.plot(x, Sigma,'b-')

ax3 = fig.add_subplot(133)
ax3.set_xlabel('feature number')
ax3.set_ylabel('Mutual information score')
ax3.plot(x, a,'r-')

plt.tight_layout()

mylist1 = list(zip(mylist, featureImportance))

mylist1 = sorted(mylist1, key = lambda x: x[1], reverse = True)
#print(mylist1)
print("doing the classification job for random forest for top 1000 ")

thelist = []
for i in range(1000):
    thelist.append(mylist1[i][0])
newTransformedData = X[:, thelist]

classificjob(newTransformedData,y)

mylist2 = list(zip(mylist, Sigma))
mylist2 = sorted(mylist2, key = lambda x: x[1], reverse = True)
#print(mylist2)
print("doing the classification job for LSA for top 1000 ")

thelist = []
for i in range(1000):
    thelist.append(mylist2[i][0])

newTransformedData = X[:, thelist]
classificjob(newTransformedData,y)


#print("the mutual information scores are as follows")
#print(a)
mylist3 = list(zip(mylist, a))
#print("the zipped value are printed ------------")
mylist3 = sorted(mylist3, key = lambda x: x[1], reverse = True)
#print(mylist3)
print("doing the classification job for mutual information score for top 1000 ")

thelist = []
for i in range(1000):
    thelist.append(mylist3[i][0])

newTransformedData = X[:, thelist]
classificjob(newTransformedData,y)

corr = pearsonr(Sigma, featureImportance)
pp = "Correlation between random forest feature and sigma value is " + str(corr)
print(pp)

corr1 = pearsonr(featureImportance, a)
pp1 = "Correlation between random forest feature importance and that of mutual information is  " + str(corr1)
print(pp1)

#fig.suptitle(pp)
#plt.bar(x, featureImportance)2
#plt.savefig('3in1-600-300_dat5.png', transparent=True, bbox_inches='tight', pad_inches=0)


"""
n = len(featureImportance)
x = range(n)

ffi_pair = zip(features, featureImportance)

ffi_pair.sort(key=lambda x: x[1])

sol = ffi_pair[::-1]
print(sol[:100])
"""

    #V = svd.components_
    #V = np.asarray(V)
    #VT = np.transpose(V)


   # print (V.shape)
   # print (VT.shape)
   # print (V[0:])
    #print (VT[:0])
    #print (np.dot(V[0:], VT[:0]))

    #print(VT[:,0])




    #print (I)
    #print(VT[0][:5])
    #print("*****************************************")
    #print (svd.components_)

    #print ("*****************************************")


    #print (U.shape)
    #print (VT.shape)

    #V = [list(x) for x in zip(*VT)]

    #print(V[0][:5])
    #print("*****************************************")
    #print(Xch[0])
    #print(svd.components_)   # it is the V matrix created term by concept matrix (each row is a concept)
    #print(svd.explained_variance_) # the variance of the row of U * S matrix row wise
    #print(svd.explained_variance_ratio_)

    #"""
    #terms = vectorizer.get_feature_names()
    #termsIncomp = []

    #for i, comp in enumerate(svd.components_) :
     #   termsIncomp = zip(terms, comp)
      #  sortedterms = sorted(termsIncomp, key =lambda x: x[1], reverse=True)[:15]
      #  print ("concept " + str(i) + " with variance " + str(Sigma[i]))
      #  for term in sortedterms:
      #      print (str(term[0]) + "  " + str(term[1]))
      #  print()

       # if i == 10:
       #     break
    #"""

    #print (sortedterms)
    #print(svd.explained_variance_.shape)

    #print ((U * Sigma).shape)

    #print (U)
    #X = lsa.fit(X)




    #X = lsa.fit_transform(X)  # if we want to perfom the dimensionality reduction

    #print (lsa.get_params(deep=True))
    #print (X.shape)
    #print("done in %fs" % (time() - t0))

    #explained_variance = svd.explained_variance_ratio_.sum()
    #print("Explained variance of the SVD step: {}%".format(
     #   int(explained_variance * 100)))

    #print()

"""
sys.stdout = old_stdout
log_file.close()

"""

###############################################################################
# Do the actual clustering

"""
if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)

print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()


if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()

    print(km.cluster_centers_ )



    print(order_centroids)

    print(len(terms))
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
"""