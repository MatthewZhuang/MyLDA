#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 12/28/2017 9:33 PM
    @desc:
        
    @author: guomianzhuang
"""
import os
import load_data
import numpy as np
from config import conf


def _idf_statistics():
    dictionary = dict()
    articles = load_data.load_articles_classify()
    articles.extend(load_data.load_articles_lda())
    print "articles number:", len(articles)
    for article in articles:
        words = article.replace("\n", "").split(" ")
        words = set(words)
        for word in words:
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1
    file_path = os.path.join(conf.PROJECT_ROOT, "data/idf_classify.txt")
    with open(file_path, "a") as f:
        for word, idf in dictionary.items():
            f.write(word+" "+str(idf)+"\n")


def load_idf():
    _idf = dict()
    file_path = os.path.join(conf.PROJECT_ROOT, "data/idf_classify.txt")
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            word, idf = line.decode("utf8").replace("\n", "").split(" ")
            _idf[word] = idf
    return _idf


def tf_idf(word_list):
    """
        normalized tfidf
    """
    N = 10000
    _idf = load_idf()
    words = word_list.replace("\n", "").split(" ")
    _tf = dict()
    for word in words:
        if word in _tf:
            _tf[word] += 1
        else:
            _tf[word] = 1
    tfidf = dict()
    max_tf = max(_tf.values())
    tf = dict()
    idf = dict()
    for word, v in _tf.items():
        tf[word] = v/float(max_tf)
    for word, v in _idf.items():
        idf[word] = np.log2(N/float(v))
    for word, v in tf.items():
        score = v*idf[word]
        tfidf[word] = score
    sum_value = sum(tfidf.values())
    for word in tfidf.keys():
        tfidf[word] /= float(sum_value)
    return tfidf


if __name__ == '__main__':
    _idf_statistics()
    # pass