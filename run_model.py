from sklearn.pipeline import Pipeline,FeatureUnion
from features import KenLMPerplexity,JaccardSimilarityAverageFeature
from read_data import load_data
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score

"""
Option to cross-validate training data
Useful to perform a bias/variance check
"""
def cross_validate(pipeline,data,cv=4):
    print "Running cross validation..."
    (Xcv,ycv) = data
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
    kick_ebos = True
    Xtrain,ytrain = load_data("train",kick_eos=kick_ebos,kick_bos=kick_ebos)
    Xdev,ydev = load_data("dev",kick_bos=kick_ebos,kick_eos=kick_ebos)
    # print KenLMPerplexity(ngram=4).transform(Xdev)
    # print KenLMPerplexity(ngram=3).transform(Xdev)
    # exit()
    nac_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('5gram_perplexity',KenLMPerplexity(ngram=5)),
            ('3gram_perplexity', KenLMPerplexity(ngram=3)),
            ('4gram_perplexity', KenLMPerplexity(ngram=4)),
            # ('jacc_sim',JaccardSimilarityAverageFeature()), #: Buggy
            # ('sent_len_mode', SentenceLengthModeFeature()), : Buggy
        ])),
        ('svm_clf',SVC(kernel='linear',C=1000,gamma=0.001))
    ])
    # cross_validate(nac_pipeline,(Xtrain,ytrain),4)
    print "Training..."
    nac_pipeline.fit(Xtrain,ytrain)
    print "Running on Development set..."
    print accuracy_score(ydev,nac_pipeline.predict(Xdev))
