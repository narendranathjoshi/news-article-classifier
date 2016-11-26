from __future__ import division

from collections import Counter

import numpy as np
from sklearn.base import TransformerMixin

import utils

"""
Use this to inspect X,y transformed values
"""
class DummyEstimator:

    def fit(self,X,y):
        print X[:5]
        return self

    def predict(self,X):
        return np.zeros(X.shape[0])

class SentenceLengthMeanFeature(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        means = []
        for article in X:
            total_length = 0
            for sentence in article:
                words = sentence.split()
                total_length += len(words)

            mean = total_length / len(X)
            means.append(mean)

        return np.array(means).reshape(len(means),1)

#Bug in this feature
class SentenceLengthModeFeature(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        modes = []
        for article in X:
            total_length = 0
            lengths = []
            for sentence in article:
                words = sentence.split()
                total_length += len(words)

            lengths.append(total_length)
            modes.append(Counter(lengths).most_common(1)[0][0])

            return np.array(modes)


class FleschKincaidReadabilityEaseFeature(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        ease_scores = []
        for article in X:
            number_of_sentences = len(article)
            number_of_words = 0
            number_of_syllables = 0
            for sentence in article:
                words = sentence.split()
                number_of_words += len(words)
                number_of_syllables += sum(map(lambda x: utils.count_syllables(x), words))

            ease_scores.append(
                utils.flesch_kincaid_ease_score(number_of_sentences, number_of_words, number_of_syllables)
            )
        return np.array(ease_scores).reshape(len(ease_scores),1)


class JaccardSimilarityAverageFeature(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        jaccard_scores = []
        for article in X:
            scores = []
            for sentence1, sentence2 in zip(article, article[1:]):
                stopped_sentence1 = utils.remove_stop_words(sentence1)
                stemmed_sentence1 = utils.stem_tokens(stopped_sentence1)
                stopped_sentence2 = utils.remove_stop_words(sentence2)
                stemmed_sentence2 = utils.stem_tokens(stopped_sentence2)

                scores.append(utils.jaccard(stemmed_sentence1, stemmed_sentence2))

            jaccard_scores.append(np.average(scores))
        return np.array(jaccard_scores).reshape(len(jaccard_scores),1)


