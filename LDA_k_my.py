#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
 @description:
        
 @Time       : 18/1/5 下午11:19
 @Author     : guomianzhuang
"""
from preprocess import load_data
# from lda_core.LdaModel import LdaModel
# from lda_core.LdaModel_SparseLDA import LdaModel
import logging.config
import time
logging.config.fileConfig("config/logger.conf")
logger = logging.getLogger("dev")

if __name__ == '__main__':
    articles = load_data.load_article_word_list()
    articles = articles[:400]
    model_id = 2
    if model_id == 1:
        logger.info("******StandardLda******")
        from lda_core.LdaModel import LdaModel
    elif model_id == 2:
        logger.info("******SparseLda******")
        from lda_core.LdaModel_SparseLDA import LdaModel
    else:
        pass

    logger.info("articles num: %d", len(articles))
    for i in range(20, 100, 20):
        begin = time.time()
        lda = LdaModel(k=i, iterations=1000, corpus=articles)
        lda.train()
        logger.info("similarity: %d", lda.inter_similarity())
        end = time.time()
        logger.info("Topic num {}, cost {}s".format(i, (end-begin)))
