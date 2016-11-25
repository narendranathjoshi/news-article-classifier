from __future__ import division

from collections import defaultdict, Counter

import numpy
from sklearn.metrics import jaccard_similarity_score

import utils

from read_data import read_data, read_labels

training = read_data("trainingSet.dat")[:500]

labels = read_labels("trainingSetLabels.dat")[:500]

REAL = '1'
FAKE = '0'


def test(X):
    jaccard_scores = []
    for article in X:
        scores = []
        for sentence1, sentence2 in zip(article, article[1:]):
            stopped_sentence1 = utils.remove_stop_words(sentence1)
            stemmed_sentence1 = utils.stem_tokens(stopped_sentence1)
            stopped_sentence2 = utils.remove_stop_words(sentence2)
            stemmed_sentence2 = utils.stem_tokens(stopped_sentence2)

            scores.append(utils.jaccard(stemmed_sentence1, stemmed_sentence2))

        jaccard_scores.append(numpy.average(scores))
    return jaccard_scores

# for i, l in zip(test(training), labels):
#     print l, i

d = defaultdict(list)
for x, l in zip(test(training), labels):
    d[l].append(x)

print "REAL"
print "Min: %f" % min(d[REAL])
print "Max: %f" % max(d[REAL])
print "Avg: %f" % numpy.average(d[REAL])
print "Median: %f" % numpy.median(d[REAL])
print "Std Dev: %f" % numpy.std(d[REAL])
print
print "FAKE"
print "Min: %f" % min(d[FAKE])
print "Max: %f" % max(d[FAKE])
print "Avg: %f" % numpy.average(d[FAKE])
print "Median: %f" % numpy.median(d[FAKE])
print "Std Dev: %f" % numpy.std(d[FAKE])
