from __future__ import division

from collections import Counter

import numpy
from sklearn.base import TransformerMixin

import utils


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

        return means


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

        return modes


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
        return ease_scores


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

            jaccard_scores.append(numpy.average(scores))
        return jaccard_scores


