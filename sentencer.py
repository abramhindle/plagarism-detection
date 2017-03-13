# Inspired by NLTK Documentation at http://www.nltk.org/_modules/nltk/tokenize/punkt.html
import nltk.data
import codecs
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
import argparse
# import matplotlib.pyplot as plt


tokenizer = RegexpTokenizer(r'\w+')

parser = argparse.ArgumentParser(description='Compare Sentences')
parser.add_argument('files', nargs=2,
                    help='files to analyze')
parser.add_argument('--min', default=0.5, help="minimum cosine similarity")
# args = parser.parse_args(['testcase-private/original.txt','testcase-private/clone.txt'])

args = parser.parse_args()

mincosine = float(args.min)
# re.split(r'[\.!]',"Hi! How are you Dr. Hindle!")
# sent_detector.tokenize("Hi! How are you Dr. Hindle!")
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
str1 = open(args.files[0]).read().decode("utf-8")
str2 = open(args.files[1]).read().decode("utf-8")
str1s = sent_detector.tokenize(str1)
str2s = sent_detector.tokenize(str2)

def stripper(x):
    return " ".join(tokenizer.tokenize(x))

strip1 = [stripper(x) for x in str1s]
strip2 = [stripper(x) for x in str2s]

vectorizer = TfidfVectorizer(min_df=1)
tfidf = vectorizer.fit_transform(strip1 + strip2)
x = (tfidf * tfidf.T).A
# hist = numpy.histogram(x)
# plt.plot(hist[1][1:],hist[0])
# plt.show

strip2s = vectorizer.transform(strip2)
strip1s = vectorizer.transform(strip1)
x = (strip1s * strip2s.T).A
y = numpy.nonzero(x >= mincosine)
print("Minimum cosine distance %s" % mincosine)
for i in range(0,len(y[0])):
    print(i)
    print("-----------------")
    print("Doc 1: %s" % strip1[y[0][i]])
    print("Doc 2: %s" % strip2[y[1][i]])
