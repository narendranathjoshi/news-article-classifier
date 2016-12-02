import numpy as np
from sklearn.metrics import accuracy_score,log_loss
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from features import *
from read_data import load_data
import math



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
    kick_ebos = True

    Xtrain, ytrain = load_data("train", kick_eos=kick_ebos, kick_bos=kick_ebos)

    Xdev, _ = load_data("test", kick_bos=kick_ebos, kick_eos=kick_ebos)

    nac_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('5gram_perplexity', KenLMPerplexity(ngram=5)),
            ('3gram_perplexity', KenLMPerplexity(ngram=3)),
            ('4gram_perplexity', KenLMPerplexity(ngram=4)),
            ('6gram_perplexity', KenLMPerplexity(ngram=6)),
            ('2gram_perplexity', KenLMPerplexity(ngram=2)),
            ('type_tokens_ratio', TypeTokenRatiosFeature()),
            
        ])),
        ('nn_mlp', MLPClassifier(hidden_layer_sizes=(100, 100,), activation='tanh', alpha=0.001, max_iter=8000,random_state=20))
    ])

    #print "Training..."
    nac_pipeline.fit(Xtrain, ytrain)
    #print "Running on Development set..."
    labels = nac_pipeline.predict(Xdev)
    probas = nac_pipeline.predict_proba(Xdev)
    
   
    
    for arr, val in zip(probas, labels):
        print "%f %f %d" % (arr[0], arr[1], val)
    


      