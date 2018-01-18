#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 12/27/2017 4:21 PM
    @desc:
        Extract keywords
        TFIDF and TextRank
    @author: guomianzhuang
"""
from TextRank import TextRank
import tf_idf
import load_data


def extract_textrank(word_list):
    tr = TextRank(word_list, 5, 0.85, 700)
    tr.cutSentence()
    tr.createNodes()
    tr.createMatrix()
    tr.calPR()
    return tr.printResult()


def extract_tfidf(word_list):
    return tf_idf.tf_idf(word_list)


def extract_ensemble(word_list, a, b, k):
    if isinstance(word_list, list):
        word_list = " ".join(word_list)
    _tfidf = extract_tfidf(word_list)
    _textrank = extract_textrank(word_list)
    _ensemble = dict()
    for key in _tfidf.keys():
        v1 = a*_tfidf[key]
        v2 = b*_textrank[key]
        _ensemble[key] = a*_tfidf[key] + b*_textrank[key]
    res = sorted(_ensemble.items(), key = lambda x : x[1], reverse=True)
    keywords = [item[0] for item in res[:k]]
    # print keywords
    # for keyword in keywords:
    #     print keyword
    return keywords


def statistics_different():
    articles = load_data.load_articles_classify()
    sim = 0
    for article in articles:
        keysets1 = extract_ensemble(article.decode("utf8"), 0, 1.0, 50)
        keysets2 = extract_ensemble(article.decode("utf8"), 1.0, 0, 50)
        keysets1 = set(keysets1)
        keysets2 = set(keysets2)
        join = keysets1.intersection(keysets2)
        union = keysets1.union(keysets2)
        sim += len(join)/float(len(union))
    print sim/len(articles)


if __name__ == '__main__':
    # articles = load_data.load_articles_classify()
    # extract_ensemble(articles[0].decode("utf8"), 0, 1.0, 50)
    statistics_different()