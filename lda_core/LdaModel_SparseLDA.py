#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
 @description:
        Lda implementation with sparse gibbs.
        The time complexity is O(MNm|NoneZero(Nkt)|).

        Use dict to save the NoneZero (k,n) and (m, k) pair.
 @Time       : 18/1/8 下午9:58
 @Author     : guomianzhuang
"""
import numpy as np
import random
import logging
import math
logger = logging.getLogger("dev")
random.seed(42)


class LdaModel:
    """
        Class ldaModel

    """
    def __init__(self, k, iterations, corpus):
        """
            :param k: topic num
            :param iterations: iter num
            :param corpus: articles , word list
        """
        self.K = k
        self.iterations = iterations
        self.beta = 0.1
        self.alpha = 50.0/k
        self.corpus = corpus
        self.dictionary = dict()          # word to index
        self.word_list = []               # index to word
        # initialize the dictionary
        for article in self.corpus:
            for word in article:
                if word not in self.dictionary:
                    index = len(self.dictionary)
                    self.dictionary[word] = index
                    self.word_list.append(word)
        self.M = len(self.corpus)   # word dictionary size, corpus size
        self.V = len(self.dictionary)
        self.nmk = np.zeros((self.M, self.K))                   # for document m, count of words labeled as k. M*K
        self.nkt = np.zeros((self.K, self.V))             # for topic k, count of words labeled as k. K*V
        self.nmk_sum = np.zeros(self.M)                 # sum for each row of nmk
        self.nkt_sum = np.zeros(self.K)                 # sum for each row of nkt
        self.hash_nmk = dict()
        self.hash_nkt = dict()
        self.phi = np.zeros((self.K, self.V))                   # topic-word distribution
        self.theta = np.zeros((self.M, self.K))                 # document-topic distribution
        # initialize the doc index
        self.doc = np.array([[0 for _ in xrange(len(self.corpus[x]))] for x in xrange(len(self.corpus))])  # document word index M*doc_size
        for m in range(self.M):
            document = self.corpus[m]
            for n in range(len(document)):
                word = document[n]
                self.doc[m][n] = self.dictionary[word]

        # topic label array
        self.z = np.array([[0 for _ in xrange(len(self.corpus[x]))] for x in xrange(len(self.corpus))])
        # random initial topic for each word
        for m in range(self.M):
            document = self.corpus[m]
            for n in range(len(document)):
                topic = int(math.floor(random.random()*self.K))
                self.z[m][n] = topic
                self.nmk[m][topic] += 1
                self.add_hash_nmk(m, topic)
                word_index = self.doc[m][n]
                self.nkt[topic][word_index] += 1
                self.add_hash_nkt(topic, word_index)
                self.nkt_sum[topic] += 1
            self.nmk_sum[m] = len(document)
        logger.info("corpus documents: %d", self.M)
        logger.info("corpus words: %d", self.V)

    def add_hash_nmk(self, m, topic):
        if m in self.hash_nmk:
            if topic in self.hash_nmk[m]:
                self.hash_nmk[m][topic] += 1
            else:
                self.hash_nmk[m][topic] = 1
        else:
            tmp = dict()
            tmp[topic] = 1
            self.hash_nmk[m] = tmp

    def add_hash_nkt(self, topic, w):
        if topic in self.hash_nkt:
            if w in self.hash_nkt[topic]:
                self.hash_nkt[topic][w] += 1
            else:
                self.hash_nkt[topic][w] = 1
        else:
            tmp = dict()
            tmp[w] = 1
            self.hash_nkt[topic] = tmp

    def minus_hash_nmk(self, m, topic):
        self.hash_nmk[m][topic] -= 1
        if self.hash_nmk[m][topic] <= 0:
            del self.hash_nmk[m][topic]
            if len(self.hash_nmk[m]) == 0:
                del self.hash_nmk[m]

    def minus_hash_nkt(self, topic, w):
        self.hash_nkt[topic][w] -= 1
        if self.hash_nkt[topic][w] <= 0:
            del self.hash_nkt[topic][w]
            if len(self.hash_nkt[topic]) == 0:
                del self.hash_nkt[topic]

    def train(self):
        """
            train the model.
        :return:
        """
        logger.info("training begin...")
        self.inference()
        logger.info("Perplexity: {}".format(self.calculate_perplexity()))
        return self.cluster()

    def inference(self):
        """
            Model inference.

            SparseLDA: three bucket.
        :return:
        """
        # initialize g and G
        g, G = self.initialize()
        for i in range(self.iterations):
            logger.info("iter %d:", i)
            for m in range(self.M):
                # initialize the c vector for current document
                c = np.zeros(self.K)
                for k in range(self.K):
                    c[k] = self.alpha / (self.nkt_sum[k] + self.beta * self.V)

                self.SparseLda(m=m, c=c, g=g)

        self.update_parameters()

    def initialize(self):
        """
            initialize the global variables.
        :return:
        """
        G = 0
        g = np.zeros(self.K)
        for k in range(self.K):
            g[k] = self.beta * self.alpha / (self.nkt_sum[k] + self.beta * self.V)
        G = sum(g)
        return g, G

    def SparseLda(self, m, c, g):
        """
            sample for each document.
        :param m:
        :param c:
        :param g:
        :return:
        """
        # updating f vector and c vector for current document
        F = 0
        f = np.zeros(self.K)
        _nmk_dict = self.hash_nmk[m]
        for _k, count in _nmk_dict.items():
            c[_k] = (self.alpha + count) / (self.nkt_sum[_k] + self.beta * self.V)
            f[_k] = self.beta * count / (self.nkt_sum[_k] + self.beta * self.V)
        F = sum(f)

        # sample a new word for each word in document m,
        N = len(self.corpus[m])
        for n in range(N):
            topic, c, g, f = self.sample_topic_gibbs_SparseLDA(m, n, c, f, g)
            self.z[m][n] = topic

    def update_parameters(self):
        """
            calculate theta and phi when sampling is done.
        :return:
        """
        for k in range(self.K):
            for n in range(self.V):
                self.phi[k][n] = (self.nkt[k][n] + self.beta)/(self.nkt_sum[k] + self.V*self.beta)
        for m in range(self.M):
            for k in range(self.K):
                self.theta[m][k] = (self.nmk[m][k] + self.alpha)/(self.nmk_sum[m] + self.M*self.alpha)

    def sample_topic_gibbs_SparseLDA(self, m, n, c, f, g):
        """
            resample the word topic.

            three bucket.
        :param m:
        :param n:
        :param c:
        :param g:
        :param f:
        :return:
        """
        old_topic = self.z[m][n]
        word_index = self.doc[m][n]

        # update g, G, f, F, c, nmk, nkt, nk for excluding zi
        self.nmk[m][old_topic] -= 1
        self.minus_hash_nmk(m, old_topic)
        self.nmk_sum[m] -= 1
        self.nkt[old_topic][word_index] -= 1
        self.minus_hash_nkt(old_topic, word_index)
        self.nkt_sum[old_topic] -= 1

        # update f F g G c
        denum = (self.nkt_sum[old_topic] + self.beta*self.V)
        c[old_topic] = (self.alpha + self.nmk[m][old_topic])/denum
        g[old_topic] = self.beta*self.alpha/denum
        f[old_topic] = self.beta*self.nmk[m][old_topic]/denum
        # calculate "topic-word" bucket E for current word
        e = np.zeros(self.K)
        for k in self.hash_nkt.keys():
            tmp = self.hash_nkt[k]
            if word_index in tmp:
                tmp_nkt = tmp[word_index]
                e[k] = tmp_nkt*c[k]
            else:
                e[k] = 0

        E = sum(e)
        G = sum(g)
        F = sum(f)
        Q = E + G + F
        # sample a new topic for current word
        U = random.random() * Q

        if U < E:
            # print("U<E")
            # sample in "topic-word" bucket  E
            topics = sorted(self.hash_nkt.keys())
            for topic in topics:
                if U < e[topic]:
                    new_topic = topic
                    break
                else:
                    U -= e[topic]
        elif U < E+F:
            # print("U<E+F")
            # sample in "document-topic" bucket  F
            U -= E
            topics = sorted(self.hash_nmk[m].keys())
            for topic in topics:
                if U < f[topic]:
                    new_topic = topic
                    break
                else:
                    U -= f[topic]
        else:
            # print("U<E+F+G")
            # sample in "smoothing only" bucket  G
            U -= (E+F)
            for topic in range(self.K):
                if U < g[topic]:
                    new_topic = topic
                    break
                else:
                    U -= g[topic]
        # update g, G, f, F, c, nmk, nkt, nk for including zi
        self.nmk[m][new_topic] += 1
        self.add_hash_nmk(m, new_topic)
        self.nmk_sum[m] += 1
        self.nkt[new_topic][word_index] += 1
        self.add_hash_nkt(new_topic, word_index)
        self.nkt_sum[new_topic] += 1

        # update f  g  c vector
        denum = (self.nkt_sum[new_topic] + self.beta * self.V)
        c[new_topic] = (self.alpha + self.nmk[m][new_topic]) / denum
        g[new_topic] = self.beta * self.alpha / denum
        f[new_topic] = self.beta * self.nmk[m][new_topic] / denum

        # reset c vector--->outside
        return new_topic, c, g, f

    def cluster(self):
        """
            label every document a topic
        :return: article-topic, M*1
        """
        logger.info("label the documents...")
        topics = np.zeros(self.M)
        for m in range(self.M):
            topic = 0
            prob = 0
            for k in range(self.K):
                tmp = self.theta[m, k]
                if tmp > prob:
                    topic = k
                    prob = tmp
            topics[m] = topic
        return topics

    def inter_similarity(self):
        """
            calculate the inter average class similarity.
        :return:
        """
        similarity = 0
        for i in range(self.K):
            for j in range(self.K):
                if i == j:
                    continue
                else:
                    a = self.phi[i]
                    b = self.phi[j]
                    num = a.dot(b)
                    denum = np.sqrt(a.dot(a))*np.sqrt(b.dot(b))
                    similarity += num/float(denum)
        return similarity/(self.K*(self.K - 1))

    def calculate_perplexity(self):
        """
            In order to save time, we calculate the perplexity of training corpus.
            perplexity(train) = exp(−∑Mm=1∑Nmn=1log∑Kk=1(φk,t)(ϑk,m)/∑Mm=1Nm)
        :return:
        """
        num = 0.0
        denum = 0.0
        for m in range(self.M):
            N = len(self.corpus[m])
            denum += N
            for n in range(N):
                tmp_sum = 0
                word_index = self.doc[m][n]
                for k in range(self.K):
                    tmp_sum += self.theta[m][k]*self.phi[k][word_index]
                log_sum = np.log(tmp_sum)
                num += log_sum
        return np.exp(-1*num/denum)


if __name__ == '__main__':
    # verify the algorithm
    articles = [["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["zhuang", "guo", "mian", "china", "ying", "Chinese"],
                ["zhuang", "guo", "mian", "china", "ying", "Chinese"],
                ["zhuang", "guo", "mian", "china", "ying", "Chinese"],
                ["zhuang", "guo", "mian", "china", "ying", "Chinese"],
                ["zhuang", "guo", "mian", "china", "ying", "Chinese"],
                ["zhuang", "guo", "mian", "china", "ying", "Chinese"],
                ["zhuang", "guo", "mian", "china", "ying", "Chinese"],
                ["zhuang", "guo", "mian", "china", "ying", "Chinese"]
                ]
    lda = LdaModel(k=2, iterations=1500, corpus=articles)
    print lda.train()
    # print lda.inter_similarity()



