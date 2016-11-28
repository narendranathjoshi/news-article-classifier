from __future__ import division

from collections import Counter

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import normalize
from read_data import load_params
import os
import utils
import kenlm

class DummyEstimator:
    """
    Use this to inspect X,y transformed values
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        print X[:5]
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

class KenLMPerplexity:

    def __init__(self,ngram):
        path_to_model = os.path.join(load_params().get("data","path"),"models","LM_{}gram.klm".format(ngram))
        self.lm = kenlm.Model(path_to_model)

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        Xtf = np.zeros(len(X))
        for i, article in enumerate(X):
            for sentence in article:
                Xtf[i] += (-self.lm.score(sentence,bos=False,eos=False))
            Xtf[i] /= len(article)
        return Xtf.reshape(Xtf.shape[0],1)







class SentenceLengthMeanFeature(TransformerMixin):
    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)

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

        return normalize(np.array(means).reshape(len(means), 1))


class SentenceLengthModeFeature(TransformerMixin):
    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)

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

        return normalize(np.array(modes).reshape(len(modes), 1))


class FleschKincaidReadabilityEaseFeature(TransformerMixin):
    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)

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
        return normalize(np.array(ease_scores).reshape(len(ease_scores), 1))


class JaccardSimilarityAverageFeature(TransformerMixin):
    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)

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

            if scores:
                jaccard_scores.append(np.average(scores))
            else:
                jaccard_scores.append(0.1)

        return normalize(np.array(jaccard_scores).reshape(len(jaccard_scores), 1))


class TypeTokenRatiosFeature(TransformerMixin):
    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        type_token_ratios = []
        for article in X:
            text = []

            for line in article:
                words = line.split(' ')
                for word in words:
                    text.append(word)

            tt_ratio = len(set(text)) / len(text)
            type_token_ratios.append(tt_ratio)

        return np.array(type_token_ratios).reshape(len(type_token_ratios), 1)


class BigramRepeatFeature(TransformerMixin):
    def __init__(self, **params):
        self.params = params

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):

        bigram_repeat = []
        for article in X:
            words = []
            count = 0
            for line in article:
                text = line.split()
                for t in text:
                    words.append(t)
            # print words
            for i in range(1, len(words)):
                denom = int(len(words) * 2)
                if '.' in words[i]:
                    continue
                if words[i - 1] == words[i]:
                    # print words[i-1],words[i]
                    count += 1
            # print count
            value = count / denom
            bigram_repeat.append(value)

        return np.array(bigram_repeat).reshape(len(bigram_repeat), 1)
