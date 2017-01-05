
from __future__ import print_function

from sklearn.feature_extraction import text
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
from nltk.stem import RSLPStemmer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import sklearn as sklearn
import sklearn.cross_validation
import numpy as np


class PorterTokenizer(object):
    def __init__(self):
        self.wnl = PorterStemmer()

    def __call__(self, doc):
        return [self.wnl.stem(t) for t in word_tokenize(doc)]


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



class StemTokenizer(object):
     def __init__(self):
         self.stm = SnowballStemmer("english")
     def __call__(self, doc):
         list1 = [self.stm.stem(t) for t in word_tokenize(doc)]
         list2 = []
         for a in list1:
             fl = 0
             for i in range(len(a)):
                 if (not(((a[i] >= 'a') and (a[i] <= 'z')) or ((a[i] >= 'A') and (a[i] <= 'Z')))) :
                     fl = 1
                     break
             if fl == 0 :
                 list2.append(a)
         #return [self.stm.stem(t) for t in word_tokenize(doc)]
             
         return list2
'''
categories = [
    'sci.space',
]
'''
stp = {"af","are","ar","aa", "aaa","ab","abl","aap","adb","zz", "think","does","just","like","said","did","the","they","our","not","a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"}

stop = text.ENGLISH_STOP_WORDS.union(stp)

def print_metrics(scores):
    """
    Compute and print evaluation metrics.
    Accuracy, Precision, Recall and F1.
    """
    print

    print("--------- Final results for Cross Validation ---------")

    print("Avg Accuracy", (np.sum(scores["accuracy"])))

    print ("Avg Precision", (np.sum(scores["precision"])))

    print("Avg Recall", (np.sum(scores["recall"])))

    print("Avg F1", (np.sum(scores["f1"])))
    print

def classificjobfortestdata(X, y):
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    ss = sklearn.cross_validation.ShuffleSplit((X.shape[0]),
                                               n_iter=2, test_size=0.1,
                                               random_state=1234)
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    print("Number of documents for training", (1 - 0.1) * len(dataset.data))

    print("Number of documents for testing", 0.1 * len(dataset.data))

    fold = 0
    for train_index, test_index in ss:
        fold += 1

        print("Fold", fold, "--------------------------------")

        # get the text and label for the training data
        train_text = X[[x for x in train_index], :]
        train_target = y[[x for x in train_index], :]

        test_text = X[[x for x in test_index], :]
        test_target = y[[x for x in test_index], :]

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

# lis = ["&&&a", "***", "an"]
# for j in range(len(lis)):
#     for i in range(len(lis[j])):
#         if (not (((a[i] >= 'a') and (a[i] <= 'z')) or ((a[i] >= 'A') and (a[i] <= 'Z')))):
#             lis.remove(a)
#             break
# # return [self.stm.stem(t) for t in word_tokenize(doc)]
# print (lis)

#dataset = fetch_20newsgroups(subset='all', categories=categories,
#                             shuffle=True, random_state=42)
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
#vectorizer2 = TfidfVectorizer(stop_words=stop, min_df = 0.018,max_df=0.50,lowercase=1,max_features=5000,tokenizer=StemTokenizer())
vectorizer2 = TfidfVectorizer(stop_words=stop,min_df=0.001,max_df=0.15,lowercase=1,tokenizer=StemTokenizer())

X2 = vectorizer2.fit_transform(dataset.data)
li2 = vectorizer2.get_feature_names()

y = dataset.target
y = np.matrix([y])
y = y.transpose()

print (dataset.filenames.shape)
classificjobfortestdata(X2, y)
print(li2)
print(len(li2))
#vectorizer1 = TfidfVectorizer(stop_words=stp,min_df=2, max_df=0.9,strip_accents='unicode',analyzer='word',lowercase=1,use_idf=1,smooth_idf=1,sublinear_tf=1,tokenizer=PorterTokenizer())

#X1 = vectorizer1.fit_transform(dataset.data)
#li1 = vectorizer1.get_feature_names()

#print(vectorizer1.get_stop_words())

#print (li1)
#print(len(li1))
#print("------------------------")



wl = WordNetLemmatizer()
sb = SnowballStemmer("english")
lc = LancasterStemmer()
ps = PorterStemmer()
rp = RSLPStemmer()

print(wl.lemmatize('characteristic'))
print(wl.lemmatize('character'))
print(wl.lemmatize('characterize'))
print(wl.lemmatize('characterized'))
print(wl.lemmatize('characterizes'))
print(wl.lemmatize('churches'))

print("-----------------------------------------------")
print(sb.stem('characteristic'))
print(sb.stem('character'))
print(sb.stem('characterize'))
print(sb.stem('characterized'))
print(sb.stem('characterizes'))

print("-----------------------------------------------")
print(lc.stem('characteristic'))
print(lc.stem('character'))
print(lc.stem('characterize'))
print(lc.stem('characterized'))
print(lc.stem('characterizes'))
print(lc.stem('christian'))
print(lc.stem('christ'))
print(lc.stem('christians'))
print(lc.stem('christify'))


print("-----------------------------------------------")
print(ps.stem('characteristic'))
print(ps.stem('character'))
print(ps.stem('characterize'))
print(ps.stem('characterized'))
print(ps.stem('characterizes'))
print(ps.stem('christian'))
print(ps.stem('christ'))
print(ps.stem('christians'))
print(ps.stem('christify'))

print("-----------------------------------------------")
print(rp.stem('characteristic'))
print(rp.stem('character'))
print(rp.stem('characterize'))
print(rp.stem('characterized'))
print(rp.stem('characterizes'))


