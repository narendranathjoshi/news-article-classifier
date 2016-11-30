from sklearn.pipeline import Pipeline,FeatureUnion
from features import *
from read_data import load_data
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
import numpy as np
from pprint import pprint
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import sys

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
            ('6gram_perplexity',KenLMPerplexity(ngram=6)),
            ('5gram_perplexity',KenLMPerplexity(ngram=5)),
            ('3gram_perplexity', KenLMPerplexity(ngram=3)),
            ('4gram_perplexity', KenLMPerplexity(ngram=4)),
            ('type_tokens_ratio', TypeTokenRatiosFeature()),
            #('bigram_repeat', BigramRepeatFeature()),
            #('flesch_kincaid_score', FleschKincaidReadabilityEaseFeature()),
            #('jacc_sim',JaccardSimilarityAverageFeature()), 
            #('sent_len_mode', SentenceLengthModeFeature()), 
        ])),
        #('svm_clf',SVC(kernel='rbf',C=100,gamma=0.001,probability = True))
        ('nn_mlp',MLPClassifier(hidden_layer_sizes=(100,100,),activation='tanh',alpha=0.001,max_iter=8000)) #93
        #('random_forest',RandomForestClassifier(n_estimators=50,criterion='entropy',max_features=int,max_depth=2))
        #('ada_boost',AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),n_estimators=200,learning_rate=0.01))
        #('svm_clf',RidgeClassifier(alpha=10))
    ])
    # cross_validate(nac_pipeline,(Xtrain,ytrain),4)
    print "Training..."
    nac_pipeline.fit(Xtrain,ytrain)
    print "Running on Development set..."
    labels = nac_pipeline.predict(Xdev)
    probas = nac_pipeline.predict_proba(Xdev)

    for arr, val, act in zip(probas, labels, ydev):
        print "%f %f %d" % (arr[0], arr[1], val)
    print accuracy_score(ydev,nac_pipeline.predict(Xdev))
