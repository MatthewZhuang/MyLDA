#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 12/27/2017 4:14 PM
    @desc:
        create corpus for LDA
    @author: guomianzhuang
"""
import os
from config import conf

source = conf.source_classify
source_original = "F:/guomianzhuang/medium/"
dest = ""


def prepare_corpus():
    """
    select 200 articles for each class,
    article words should be more than 200?
    :return:
    """
    dirs = os.listdir(source)
    for dir in dirs:
        path = os.path.join(source, dir)
        files = os.listdir(path)
        file_num = 0
        for file in files:
            file_path = os.path.join(path, file)
            with open(file_path) as f:
                content = f.readline().split(" ")
            if (len(content) > 200) and (file_num < 200):
                file_num += 1
            else:
                os.remove(file_path)


def statistics_corpus():
    """
    select 200 articles for each class,
    article words should be more than 200?
    :return:
    """
    dirs = os.listdir(source)
    for dir in dirs:
        path = os.path.join(source, dir)
        files = os.listdir(path)
        file_num = 0
        word_num = 0
        word_len_100 = 0
        for file in files:
            file_num += 1
            file_path = os.path.join(path, file)
            with open(file_path) as f:
                content = f.readline().split(" ")
                word_num += len(content)
                if len(content) > 200:
                    word_len_100 += 1
        print("class %s has avg word num %d" % (dir, float(word_num)/file_num))
        print("class %s has %d articles len > 200" % (dir, word_len_100))


def filter_data():
    dirs = os.listdir(source)
    dizhen = []
    for dir in dirs:
        print dir
        path = os.path.join(source, dir)
        files = os.listdir(path)
        # if dir != "lvyou":
        #     continue
        for file in files:
            file_path = os.path.join(path, file)
            tmp = ""
            with open(file_path) as f:
                content = f.readline()
                if ("地震" in content) or ("抗震救灾" in content) :
                    tmp = content
            if len(tmp) > 0:
                # with open("F:/guomianzhuang/LDA/classify_corpus/dizhen/"+file, mode="w") as f:
                #     f.write(tmp)
                print file
                os.remove(file_path)
    print dizhen


def generate_class_data():
    dirs = os.listdir(source_original)
    for dir in dirs:
        print dir
        path = os.path.join(source_original, dir)
        files = os.listdir(path)
        if dir != "yule":
            continue
        for file in files:
            file_path = os.path.join(path, file)
            tmp = ""
            with open(file_path) as f:
                content = f.readline()
                save = content[:150]
                if ("电影" in save) or ("电视剧" in save) :
                    tmp = content
            if len(tmp) > 0:
                with open("F:/guomianzhuang/LDA/classify_corpus_topic/yule/"+file, mode="w") as f:
                    f.write(tmp)


if __name__ == '__main__':
    generate_class_data()
    # filter_data()