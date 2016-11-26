from sklearn.pipeline import Pipeline, FeatureUnion
from features import FleschKincaidReadabilityEaseFeature, SentenceLengthMeanFeature, \
    JaccardSimilarityAverageFeature, SentenceLengthModeFeature, DummyEstimator
from read_data import load_data
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score

"""
Option to cross-validate training data
Useful to perform a bias/variance check
"""


def cross_validate(pipeline, data, cv=4):
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
    nac_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('sentence_length_mean', SentenceLengthMeanFeature()),
            ('flesch_kincaid_score', FleschKincaidReadabilityEaseFeature()),
            ('jaccard_similarity',JaccardSimilarityAverageFeature()),
            ('sentence_length_mode', SentenceLengthModeFeature()),
        ])),
        ('logreg_clf', LogisticRegression())
    ])

    # cross_validate(nac_pipeline,(Xtrain,ytrain),4)
    print "Training..."
    nac_pipeline.fit(Xtrain, ytrain)

    print "Running on Development set..."
    print accuracy_score(ydev, nac_pipeline.predict(Xdev))
