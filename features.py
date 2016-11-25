from __future__ import division

from collections import Counter

from sklearn.base import TransformerMixin

import utils


class SentenceLengthMeanFeature(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        means = []
        total_length = 0
        for article in X:
            for sentence in article:
                total_length += len(sentence)

            mean = total_length / len(X)
            means.append(mean)

        return means


class SentenceLengthModeFeature(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        modes = []
        total_length = 0
        for article in X:
            lengths = []
            for sentence in article:
                total_length += len(sentence)

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
                number_of_words += len(sentence)
                number_of_syllables += sum(map(lambda x: utils.count_syllables(x), sentence))

            ease_scores.append(
                utils.flesch_kincaid_ease_score(number_of_sentences, number_of_words, number_of_syllables)
            )
        return ease_scores


