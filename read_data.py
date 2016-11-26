import json
import os
import numpy as np

def load_params():
    """
    Load parameters from settings.json
    :return:set of parameters loaded from settings.json
    """
    json_data = open("settings.json")
    params = json.load(json_data)
    return params

def load_data(data_type):
    """Function that loads data according to the data_type: train/test/cv"""
    assert(data_type == "train" or data_type == "test" or data_type == "dev")
    base_path = load_params()["data_src"]
    if data_type == "train":
        TRAIN_DAT_FILE = os.path.join(base_path,"trainingSet.dat.txt")
        TRAIN_LABELS_FILE = os.path.join(base_path,"trainingSetLabels.dat.txt")
        X = read_data(TRAIN_DAT_FILE)
        y = read_labels(TRAIN_LABELS_FILE)
        return X,y
    elif data_type == "dev":
        DEV_DAT_FILE = os.path.join(base_path, "developmentSet.dat.txt")
        DEV_LABELS_FILE = os.path.join(base_path, "developmentSetLabels.dat")
        X = read_data(DEV_DAT_FILE)
        y = read_labels(DEV_LABELS_FILE)
        return X,y
    else:
        # Need to decide about handling the testing data
        pass

def read_data(filename):
    text = open(filename).read().split("~~~~~")
    text = [line.strip().split("\n") for line in filter(None, text)]
    return np.array(text)


def read_labels(filename):
    text = map(int,open(filename).read().splitlines())
    return np.array(text)
