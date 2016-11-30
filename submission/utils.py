from __future__ import division
import re
from stop_words import get_stop_words
from nltk.stem import PorterStemmer

# Get stopwords for English
en_stop = get_stop_words('en')


def count_syllables(word):
    """
    Count the number of syllables in a given word
    :param word:
    :return:
    """
    word = word.lower()

    exception_add = ['serious', 'crucial']
    exception_del = ['fortunately', 'unfortunately']

    co_one = ['cool', 'coach', 'coat', 'coal', 'count', 'coin', 'coarse', 'coup', 'coif', 'cook', 'coign', 'coiffe',
              'coof', 'court']
    co_two = ['coapt', 'coed', 'coinci']

    pre_one = ['preach']

    syls = 0  # added syllable number
    disc = 0  # discarded syllable number

    if len(word) <= 3:
        syls = 1
        return syls

    if word[-2:] == "es" or word[-2:] == "ed":
        doubleAndtripple_1 = len(re.findall(r'[eaoui][eaoui]', word))
        if doubleAndtripple_1 > 1 or len(re.findall(r'[eaoui][^eaoui]', word)) > 1:
            if word[-3:] == "ted" or word[-3:] == "tes" or word[-3:] == "ses" or word[-3:] == "ied" or word[
                                                                                                       -3:] == "ies":
                pass
            else:
                disc += 1

    le_except = ['whole', 'mobile', 'pole', 'male', 'female', 'hale', 'pale', 'tale', 'sale', 'aisle', 'whale', 'while']

    if word[-1:] == "e":
        if word[-2:] == "le" and word not in le_except:
            pass

        else:
            disc += 1

    doubleAndtripple = len(re.findall(r'[eaoui][eaoui]', word))
    tripple = len(re.findall(r'[eaoui][eaoui][eaoui]', word))
    disc += doubleAndtripple + tripple

    numVowels = len(re.findall(r'[eaoui]', word))

    if word[:2] == "mc":
        syls += 1

    if word[-1:] == "y" and word[-2] not in "aeoui":
        syls += 1

    for i, j in enumerate(word):
        if j == "y":
            if (i != 0) and (i != len(word) - 1):
                if word[i - 1] not in "aeoui" and word[i + 1] not in "aeoui":
                    syls += 1

    if word[:3] == "tri" and word[3] in "aeoui":
        syls += 1

    if word[:2] == "bi" and word[2] in "aeoui":
        syls += 1

    if word[-3:] == "ian":
        if word[-4:] == "cian" or word[-4:] == "tian":
            pass
        else:
            syls += 1

    if word[:2] == "co" and word[2] in 'eaoui':

        if word[:4] in co_two or word[:5] in co_two or word[:6] in co_two:
            syls += 1
        elif word[:4] in co_one or word[:5] in co_one or word[:6] in co_one:
            pass
        else:
            syls += 1

    if word[:3] == "pre" and word[3] in 'eaoui':
        if word[:6] in pre_one:
            pass
        else:
            syls += 1

    negative = ["doesn't", "isn't", "shouldn't", "couldn't", "wouldn't"]

    if word[-3:] == "n't":
        if word in negative:
            syls += 1
        else:
            pass

    if word in exception_del:
        disc += 1

    if word in exception_add:
        syls += 1

    return numVowels - disc + syls


def flesch_kincaid_ease_score(number_of_sentences, number_of_words, number_of_syllables):
    """
    Compute the Flesch-Kincaid ease of readability score for a given article

    :param number_of_sentences:
    :param number_of_words:
    :param number_of_syllables:
    :return:
    """
    score = 206.835
    score -= (1.015 * (number_of_words / number_of_sentences))
    score -= (84.6 * (number_of_syllables / number_of_words))
    return score


def remove_stop_words(sentence):
    """
    Remove stop words from the sentence (a string) and returns the sentence as a (list of words)

    :param sentence:
    :return:
    """
    global en_stop
    tokens = sentence.replace("<s>", "").replace("</s>", "").strip().split()
    stopped_string = [i for i in tokens if not i in en_stop]
    return stopped_string


def stem_tokens(tokens):
    """
    Stem the tokens (a list of strings) and returns a (list of stems)

    :param tokens:
    :return:
    """
    stemmed = []
    for item in tokens:
        stemmed.append(PorterStemmer().stem(item))
    return stemmed


def jaccard(a, b):
    """
    Compute the Jaccard Similarity between two sentences a and b.
    a and b are (list of stems)

    :param a:
    :param b:
    :return:
    """
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return len(c) / (len(a) + len(b) - len(c))
