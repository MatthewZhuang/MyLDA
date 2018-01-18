#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 12/29/2017 2:39 PM
    @desc:
        learn and inference
    @author: guomianzhuang
"""
import os
import json
from gensim import corpora
from gensim import models
import numpy as np
from preprocess import load_data
from config import conf
PROJECT_ROOT = conf.PROJECT_ROOT
m_path = os.path.join(PROJECT_ROOT, "data/{0}_model.m".format('lda'))
m50_path = os.path.join(PROJECT_ROOT, "data/{0}_model.m".format('lda50'))
m80_path = os.path.join(PROJECT_ROOT, "data/{0}_model.m".format('lda80'))


def generateDict():
    d_path = os.path.join(PROJECT_ROOT, "data/lda_dict.dict")
    corpus = load_data.load_article_word_list()
    print len(corpus)
    dictionary = corpora.Dictionary(corpus)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
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


def calc_perplexity(m, c):
    import numpy as np
    return np.exp(-m.log_perplexity(c))


def trainModel(training, num, dictionary, whe, key):
    if key == 50:
        save_path = m50_path
    elif key == 80:
        save_path = m80_path
    else:
        save_path = m_path
    if whe == 1:
        print "train model begin."
        selected = None
        p_value = 10000000
        for pass_num in [40]:
            m = models.LdaModel(corpus=training, id2word=dictionary, num_topics=num, iterations=500, alpha='symmetric',
                                passes=pass_num, random_state=42)
            p1 = calc_perplexity(m, training)
            print("{0}: perplexity is {1}".format(50, p1))
            if p1 < p_value:
                selected = m
        selected.save(save_path)
        print "train model end."
    m = models.LdaModel.load(save_path)
    return m


def load_corpus(k):
    articles_dict = load_data.load_articles_with_class(k)
    texts = []
    for key in articles_dict.keys():
        texts.extend(articles_dict[key])
    dictionary = loadDict('lda')
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus, texts, articles_dict


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


def run():
    #############generate corpus###################
    key = 100
    whe_train = 0
    dictionary = loadDict("lda")
    corpus, texts, articles_dict = load_corpus(key)
    print len(corpus)
    #############trian the model###################
    topic_num = 7
    model = trainModel(corpus, topic_num, dictionary, whe_train, key)
    rs = model.print_topics(20)
    for record in rs:
        print record[1].encode("utf8")
    ##############predict new sample##############
    true_labeled = dict()
    for key in articles_dict.keys():
        print key
        predict_label = dict()
        articles = articles_dict[key]
        for i in range(len(articles)):
            # print " ".join([word.decode("utf8") for word in articles[i]]).encode("utf8")
            label = predict(articles[i], model)
            if label in predict_label.keys():
                predict_label[label] += 1
            else:
                predict_label[label] = 1
        total = sum(predict_label.values())
        print "total", total
        right = max(predict_label.values())

        print "right", right
        print("key %s recall: %f", (key, (float(right)/float(total))))

if __name__ == '__main__':
    run()

