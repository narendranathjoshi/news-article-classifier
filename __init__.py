from pprint import pprint
from time import time

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from features import FleschKincaidReadabilityEaseFeature, SentenceLengthMeanFeature, \
    JaccardSimilarityAverageFeature, SentenceLengthModeFeature, DummyEstimator, TypeTokenRatiosFeature, \
    BigramRepeatFeature
from read_data import load_data
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


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
    Xtrain, ytrain = load_data("train")
    Xdev, ydev = load_data("dev")

    params = {
        "svm": {
            'classifier__C': [1, 10, 100, 1000],
            'classifier__gamma': [0.001, 0.0001]
        },
        "nn_mlp": {
            "hidden_layer_sizes": (100, 100,),
            "activation": 'logistic',
            "alpha": 0.0001,
            "max_iter": 1000
        }
    }

    classifiers = {
        "svm": SVC(),
        "logreg": LogisticRegression(),
        "sgd": SGDClassifier(),
        "nn_mlp": MLPClassifier(),
    }

    CLASSIFIER = "svm"

    nac_pipeline = Pipeline([
        ('features', FeatureUnion([
            # ('sentence_length_mean', SentenceLengthMeanFeature()),
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
