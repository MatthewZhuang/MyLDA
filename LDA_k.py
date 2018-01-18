#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 1/3/2018 7:46 PM
    @desc:
        
    @author: guomianzhuang
"""
import os
import json
from gensim import corpora
from gensim import models
import numpy as np
from preprocess import load_data
from config import conf
import numpy as np
from numpy import linalg
PROJECT_ROOT = conf.PROJECT_ROOT
m_path = os.path.join(PROJECT_ROOT, "data/{0}_model.m".format('lda'))


def generateDict():
    d_path = os.path.join(PROJECT_ROOT, "data/lda_k_dict.dict")
    corpus = load_data.load_article_word_list()
    dictionary = corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_below=3, no_above=0.7)
    print len(dictionary.values())
    dictionary.save(d_path)

# generateDict()
# exit(0)


def loadDict(p):
    d_path = os.path.join(PROJECT_ROOT, "data/{0}_dict.dict".format(p))
    d = corpora.Dictionary.load(d_path)
    return d


def loadCorpus(p):
    c_path = os.path.join(PROJECT_ROOT, "data/{0}_corpus.mm".format(p))
    c = corpora.MmCorpus(c_path)
    return c


def load_corpus():
    articles = load_data.load_article_word_list()
    dictionary = loadDict('lda_k')
    corpus = [dictionary.doc2bow(text) for text in articles]
    return corpus, articles


def predict(word_list, m):
    # m = models.LdaModel.load(m_path)
    dictionary = loadDict('lda')
    doc_bow = dictionary.doc2bow(word_list)      #文档转换成bow
    doc_lda = m[doc_bow]                   #得到新文档的主题分布
    #输出新文档的主题分布
    # print doc_lda
    mi = 0
    l = 0
    for label, score in doc_lda:
        if score > mi:
            l = label
            mi = score
    # print l
    return l


def cal_sim(m, k):
    sim = 0
    topics = []
    for i in range(k):
        topic = m.get_topic_terms(i, 40000)
        topics.append(topic)
    for topic_i in topics:
        for topic_j in topics:
            if topic_i == topic_j:
                continue
            sim+=similarity(topic_i, topic_j)
    return sim/float(k*(k-1))


def similarity(list1, list2):
    v1 = [0.0]*40000
    v2 = [0.0]*40000
    for id, value in list1:
        v1[id] = value
    for id, value in list2:
        v2[id] = value
    v1 = np.array(v1)
    v2 = np.array(v2)
    num = v1.dot(v2)
    denom = np.sqrt(v1.dot(v1)) * np.sqrt(v2.dot(v2))
    cos = num / denom #余弦值
    return cos


def trainModel(training, dictionary, whe):
    save_path = m_path
    if whe == 1:
        print "train model begin."
        selected = None
        sim = 10000
        for k in range(10, 300, 30):
            m = models.LdaModel(corpus=training, id2word=dictionary, num_topics=k, iterations=500, alpha='symmetric',
                                passes=40, random_state=42)
            tmp = cal_sim(m, k)
            print k, tmp
            if tmp < sim:
                sim = tmp
                selected = m
        selected.save(save_path)
        selected.print_topic(20)
        print "train model end."
    m = models.LdaModel.load(save_path)
    return m


def run():
    whe_train = 1
    #############generate corpus###################
    dictionary = loadDict("lda_k")
    corpus, texts = load_corpus()
    print len(corpus)
    #############trian the model###################
    model = trainModel(corpus, dictionary, whe_train)
    # rs = model.print_topics(20)
    # for record in rs:
    #     print record[1].encode("utf8")
    ##############predict new sample##############


if __name__ == '__main__':
    run()
    # pass

