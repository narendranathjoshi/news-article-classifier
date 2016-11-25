from __future__ import division
from sklearn.base import TransformerMixin


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


