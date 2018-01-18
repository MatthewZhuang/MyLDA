#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 12/28/2017 9:27 PM
    @desc:
        load articles
    @author: guomianzhuang
"""
import os
import keywords
from config import conf
source_lda = conf.source_lda
source_classify = conf.source_classify


def load_articles_lda():
    file_list = os.listdir(source_lda)
    result = []
    for file_name in file_list:
        file_path = os.path.join(source_lda, file_name)
        with open(file_path) as f:
            articles = f.readlines()
            for article in articles:
                article = article.replace("\n", "")
                result.append(article)
    return result


def load_articles_classify():
    dir_list = os.listdir(source_classify)
    result = []
    for dir_name in dir_list:
        path = os.path.join(source_classify, dir_name)
        if not os.path.isdir(path):
            continue
        file_list = os.listdir(path)
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            with open(file_path) as f:
                articles = f.readlines()
                for article in articles:
                    article = article.replace("\n", "")
                    result.append(article)
    return result


def load_articles_with_class(k):
    if k == 50:
        dir_list = os.listdir(conf.source_classify50)
    else:
        dir_list = os.listdir(source_classify)
    article_labeled = dict()
    for dir_name in dir_list:
        path = os.path.join(source_classify, dir_name)
        if not os.path.isdir(path):
            print path
            continue
        file_list = os.listdir(path)
        _articles = []
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            with open(file_path) as f:
                articles = f.readlines()
                for article in articles:
                    article = article.replace("\n", "").split(" ")
                    _articles.append(article)
        print("%s class : %d" % (dir_name, len(_articles)))
        article_labeled[dir_name] = _articles
    return article_labeled


def load_article_word_list():
    articles = load_articles_classify()
    # articles2 = load_articles_lda()
    # articles.extend(articles2)
    result = []
    for article in articles:
        result.append(article.split(" "))
    return result


def transform_corpus(articles, k):
    result = []
    for article in articles:
        article_string = " ".join(article)
        keys = keywords.extract_ensemble(article_string.decode("utf8"), 0, 1.0, k)
        temp = [word for word in article if word.decode("utf8") in keys]
        result.append(temp)
    return result


def generate_corpus_key(k):
    article_labeled = load_articles_with_class(0)
    file_to_write = None
    if k == 50:
        file_to_write = conf.source_classify50
    if k == 80:
        file_to_write = conf.source_classify80
    for path in article_labeled.keys():
        file_path = os.path.join(file_to_write, path, "1.txt")
        articles = article_labeled[path]
        articles = transform_corpus(articles, k)
        with open(file_path, "a") as f:
            for article in articles:
                f.write(" ".join(article)+"\n")


if __name__ == '__main__':
    generate_corpus_key(80)

