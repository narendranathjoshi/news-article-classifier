import json
import os
from ConfigParser import SafeConfigParser

import numpy as np


def load_params():
    """
    Load parameters from settings.ini

    :return:set of parameters loaded from settings.ini
    """
    parser = SafeConfigParser()
    parser.read('settings.ini')
    return parser


def load_data(data_type):
    """
    Function that loads data according to the data_type: train/test/cv

    :param data_type:
    :return:
    """
    assert (data_type == "train" or data_type == "test" or data_type == "dev")
    parser = load_params()
    base_path = parser.get("data", "path")

    if data_type == "train":
        TRAIN_DAT_FILE = os.path.join(base_path, parser.get("training", "data"))
        TRAIN_LABELS_FILE = os.path.join(base_path, parser.get("training", "labels"))
        X = read_data(TRAIN_DAT_FILE)
        y = read_labels(TRAIN_LABELS_FILE)
        return X, y

    elif data_type == "dev":
        DEV_DAT_FILE = os.path.join(base_path, parser.get("development", "data"))
        DEV_LABELS_FILE = os.path.join(base_path, parser.get("development", "labels"))
        X = read_data(DEV_DAT_FILE)
        y = read_labels(DEV_LABELS_FILE)
        return X, y

    else:
        # Need to decide about handling the testing data
        pass


def read_data(filename):
    text = open(filename).read().split("~~~~~")
    text = [line.strip().split("\n") for line in filter(None, text)]
    return np.array(text)


def read_labels(filename):
    text = map(int, open(filename).read().splitlines())
    return np.array(text)
