from pprint import pprint
from time import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC

from features import *
from read_data import load_data


def cross_validate(pipeline, data, cv=4):
    """
    Option to cross-validate training data
    Useful to perform a bias/variance check

    :param pipeline:
    :param data:
    :param cv:
    :return:
    """
    print "Running cross validation..."
    (Xcv, ycv) = data
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    results = []
    for train_idx, val_idx in kfold.split(Xtrain):
        pipeline.fit(Xcv[train_idx], ycv[train_idx])
        results.append(accuracy_score(
            ycv[val_idx], pipeline.predict(Xcv[val_idx])
        ))
    print "{} +/- {}".format(np.mean(results), np.std(results))


if __name__ == '__main__':
    print "Importing data..."
    Xtrain, ytrain = load_data("train",kick_eos=True,kick_bos=True)
    Xdev, ydev = load_data("dev",kick_bos=True,kick_eos=True)
    params = {
        "svm": {
            'classifier__C': [1000, 2000],
            'classifier__gamma': [0.001, 0.005]
        },
        "nn_mlp": {
            "classifier__hidden_layer_sizes": [(100,), (100, 100,)],
            "classifier__activation": ['tanh', 'logistic'],
            "classifier__alpha": [0.001],
            "classifier__max_iter": [5000, 8000, 10000]
        }
    }

    classifiers = {
        "svm": SVC(),
        "logreg": LogisticRegression(),
        "sgd": SGDClassifier(),
        "nn_mlp": MLPClassifier(),
    }

    CLASSIFIER = "nn_mlp"

    nac_pipeline = Pipeline([
        ('features', FeatureUnion([
            # ('sentence_length_mean', SentenceLengthMeanFeature()),
            ('6gram_perplexity', KenLMPerplexity(ngram=6)),
            ('5gram_perplexity', KenLMPerplexity(ngram=5)),
            ('3gram_perplexity', KenLMPerplexity(ngram=3)),
            ('4gram_perplexity', KenLMPerplexity(ngram=4)),
            ('sentence_length_mode', SentenceLengthModeFeature()),
            ('flesch_kincaid_score', FleschKincaidReadabilityEaseFeature()),
            ('jaccard_similarity', JaccardSimilarityAverageFeature()),
            ('type_tokens_ratio', TypeTokenRatiosFeature()),
            ('bigram_repeat', BigramRepeatFeature()),
        ])),
        ('classifier', classifiers[CLASSIFIER]),
    ])

    # cross_validate(nac_pipeline,(Xtrain,ytrain),4)
    # print "Training..."
    # pipeline = nac_pipeline.fit(Xtrain, ytrain)
    #
    # print "Running on Development set..."
    # print accuracy_score(ydev, nac_pipeline.predict(Xdev))

    grid_search = GridSearchCV(nac_pipeline, param_grid=params[CLASSIFIER], n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("nac_pipeline:", [name for name, _ in nac_pipeline.steps])
    print("parameters:")
    pprint(params[CLASSIFIER])
    t0 = time()
    grid_search.fit(Xtrain, ytrain)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params[CLASSIFIER].keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
