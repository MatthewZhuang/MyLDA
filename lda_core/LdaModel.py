#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 1/5/2018 1:13 PM
    @desc:
        LDA model implementation with  standard Gibbs.
        Time Complexity: O(M*AVG(Nm)*K)
        This is too slow to real applications.
    @author: guomianzhuang
"""
import numpy as np
import random
import logging
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
        self.phi = np.zeros((self.K, self.V))                   # topic-word distribution
        self.theta = np.zeros((self.M, self.K))                 # document-topic distribution
        # initialize the doc index
        self.doc = np.array([[0 for _ in xrange(len(self.corpus[x]))] for x in xrange(len(self.corpus))])
                                            # document word index M*doc_size
        for i in range(self.M):
            document = self.corpus[i]
            for j in range(len(document)):
                word = document[j]
                self.doc[i][j] = self.dictionary[word]

        # topic label array
        self.z = np.array([[0 for _ in xrange(len(self.corpus[x]))] for x in xrange(len(self.corpus))])
        # random initial topic for each word
        for m in range(self.M):
            document = self.corpus[m]
            for n in range(len(document)):
                topic = int(random.random()*self.K)
                self.z[m][n] = topic
                self.nmk[m][topic] += 1
                word_index = self.doc[m][n]
                self.nkt[topic][word_index] += 1
                self.nkt_sum[topic] += 1
            self.nmk_sum[m] = len(document)
        logger.info("corpus documents: %d", self.M)
        logger.info("corpus words: %d", self.V)

    def train(self):
        logger.info("training begin...")
        self.inference()
        print self.calculate_perplexity()
        return self.cluster()

    def inference(self):
        """
            Model inference.
        :return:
        """
        for i in range(self.iterations):
            logger.info("iter %d:", i)
            for m in range(self.M):
                N = len(self.corpus[m])
                for n in range(N):
                    topic = self.sample_topic_gibbs(m, n)
                    self.z[m][n] = topic
        self.update_parameters()

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

    def sample_topic_gibbs(self, m, n):
        """
            resample the word topic.
        :param m:
        :param n:
        :return:
        """
        old_topic = self.z[m][n]
        self.nmk[m][old_topic] -= 1
        self.nmk_sum[m] -= 1
        word_index = self.doc[m][n]
        self.nkt[old_topic][word_index] -= 1
        self.nkt_sum[old_topic] -= 1
        prob = np.zeros(self.K)
        for k in range(self.K):
            prob[k] = ((self.nkt[k][word_index] + self.beta)/(self.nkt_sum[k] + self.V*self.beta)) \
                * ((self.nmk[m][k] + self.alpha)/(self.nmk_sum[m] + self.K*self.alpha))
        for i in range(1, self.K):
            prob[i] += prob[i-1]
        position = random.random()*prob[self.K - 1]

        new_topic = old_topic
        for i in range(self.K):
            if position < prob[i]:
                new_topic = i
                break
        self.nmk[m][new_topic] += 1
        self.nmk_sum[m] += 1
        self.nkt[new_topic][word_index] += 1
        self.nkt_sum[new_topic] += 1
        return new_topic

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
    articles = [["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["hello", "world", "is", "a", "good", "pad"],
                ["zhuang", "guo", "mian", "heihei", "ying", "huawei"],
                ["zhuang", "guo", "mian", "heihei", "ying", "huawei"],
                ["zhuang", "guo", "mian", "heihei", "ying", "huawei"],
                ["zhuang", "guo", "mian", "heihei", "ying", "huawei"],
                ["zhuang", "guo", "mian", "heihei", "ying", "huawei"],
                ["zhuang", "guo", "mian", "heihei", "ying", "huawei"],
                ["zhuang", "guo", "mian", "heihei", "ying", "huawei"],
                ["zhuang", "guo", "mian", "heihei", "ying", "huawei"]
                ]
    lda = LdaModel(k=2, iterations=1500, corpus=articles)
    print lda.train()
    # print lda.inter_similarity()
    


