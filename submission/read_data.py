import os
import sys
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


def load_data(data_type, kick_bos=False, kick_eos=False):
    """
    Function that loads data according to the data_type: train/test/cv

    :param kick_eos:
    :param kick_bos:
    :param data_type:
    :return:
    """
    assert (data_type == "train" or data_type == "test" or data_type == "dev")
    parser = load_params()
    base_path = parser.get("data", "path")

    if data_type == "train":
        TRAIN_DAT_FILE = os.path.join(base_path, parser.get("training", "data"))
        TRAIN_LABELS_FILE = os.path.join(base_path, parser.get("training", "labels"))
        X = read_data(TRAIN_DAT_FILE, kick_bos, kick_eos)
        y = read_labels(TRAIN_LABELS_FILE)
        return X, y

    elif data_type == "dev":
        DEV_DAT_FILE = os.path.join(base_path, parser.get("development", "data"))
        DEV_LABELS_FILE = os.path.join(base_path, parser.get("development", "labels"))
        X = read_data(DEV_DAT_FILE, kick_bos, kick_eos)
        y = read_labels(DEV_LABELS_FILE)
        return X, y

    elif data_type == "test":
        X = read_data(kick_bos=kick_bos, kick_eos=kick_eos, stdin=True)
        return X, None


def kick_bos_eos(l, kick_bos=False, kick_eos=False):
    """
    Remove beginning of sentence and end of sentence tags

    :param l:
    :param kick_bos:
    :param kick_eos:
    :return:
    """
    if kick_bos:
        l = l.strip("<s> ")
    if kick_eos:
        l = l.strip(" </s>")
    return l


def read_data(filename=None, kick_bos=False, kick_eos=False, stdin=False):
    """
    Read data from file or standard input

    :param filename:
    :param kick_bos:
    :param kick_eos:
    :param stdin:
    :return:
    """
    if stdin:
        text = ""
        for line in sys.stdin:
            text += line
        text = text.split("~~~~~")
    else:
        text = open(filename).read().split("~~~~~")

    text = [map(lambda l: kick_bos_eos(l, kick_bos=kick_bos, kick_eos=kick_eos), line.strip().split("\n")) for line in
            filter(None, text)]
    return np.array(text)


def read_labels(filename):
    """
    Read label files

    :param filename:
    :return:
    """
    text = map(int, open(filename).read().splitlines())
    return np.array(text)
