#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 12/27/2017 8:22 PM
    @desc:
        
    @author: guomianzhuang
"""
import random
source = "F:/guomianzhuang/concept/cptall.txt"


def sta_cpt():
    cpts = dict()
    with open(source) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\n", "")
            if line in cpts:
                cpts[line] += 1
            else:
                cpts[line] = 1
    print len(cpts)


def generate_cpt_corpus():
    sentences = []
    with open("F:/guomianzhuang/concept/train.txt") as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace(" ", "").replace("\n", "。\n")
            if (len(line.decode("utf8")) > 20) and \
                    (line not in sentences) and\
                    (("概念" in line) or ("庄郭冕涨" in line)):
                sentences.append(line)
    with open("F:/guomianzhuang/concept/train_filter.txt", "a") as f:
        for sentence in sentences:
            f.write(sentence)


def load_cpts():
    with open("F:/guomianzhuang/concept/cpt_all.txt") as f:
        lines = f.readlines()
    return [cpt.replace("\n", "") for cpt in lines]


def sta_cpt():
    cpts = load_cpts()
    with open("F:/guomianzhuang/concept/train_filter.txt") as f:
        lines = f.readlines()
    sta = dict()
    for line in lines:
        for cpt in cpts:
            if cpt in line:
                if cpt in sta:
                    sta[cpt] += 1
                else:
                    sta[cpt] = 1
    for w, c in sta.items():
        print w, c


def generate_cpt_corpus(count):
    """
        generate corpus for cpt recognition
        count sentence for each cpt.
    """
    selected = []
    cpts = load_cpts()
    with open("F:/guomianzhuang/concept/train_filter.txt") as f:
        lines = f.readlines()
    sta = dict()
    for line in lines:
        for cpt in cpts:
            if cpt in line:
                if cpt in sta:
                    if sta[cpt] <= count:
                        selected.append(line)
                        sta[cpt] += 1
                else:
                    selected.append(line)
                    sta[cpt] = 1
                break

    for w in cpts:
        c = sta[w] if w in sta else 0
        if c < 10:
            num = 10 - c
            for i in range(num):
                sen = random.choice(lines)
                sen = random_replace(cpts, w, sen)
                selected.append(sen)

    with open("F:/guomianzhuang/concept/train_select.txt", "a") as f:
        for sen in selected:
            f.write(sen)


def random_replace(cpts, cpt, sentence):
    for c in cpts:
        if c in sentence:
            sentence = sentence.replace(c, cpt)
            return sentence


def label_train():
    cpts = load_cpts()
    with open("F:/guomianzhuang/concept/train_select.txt") as f:
        lines = f.readlines()
    labeled = []
    for line in lines:
        line = line.replace("\n", "").decode("UTF8")
        length = len(line)
        labels = ["O"]*len(line)
        for cpt in cpts:
            cpt = cpt.decode("utf8")
            if cpt in line:
                index = line.index(cpt)
                for i in range(len(cpt)):
                    if i == 0:
                        labels[index] = "B-CPT"
                    else:
                        labels[index+i] = "I-CPT"
        temp = []
        for i in range(length):
            temp.append(line[i])
            temp.append(labels[i])

        labeled.append("".join(temp))

    with open("F:/guomianzhuang/concept/labeled.txt", "a") as f:
        for l in labeled:
            f.write(l.encode("utf8")+"\n")


def cpt_label_train():
    """
        standard label format
    """
    cpts = load_cpts()
    with open("F:/guomianzhuang/concept/train_select.txt") as f:
        lines = f.readlines()
    for line in lines:
        line = line.replace("\n", "").decode("UTF8")
        length = len(line)
        labels = ["O"]*len(line)
        for cpt in cpts:
            cpt = cpt.decode("utf8")
            if cpt in line:
                index = line.index(cpt)
                for i in range(len(cpt)):
                    if i == 0:
                        labels[index] = "B-CPT"
                    else:
                        labels[index+i] = "I-CPT"

        with open("F:/guomianzhuang/concept/cpt.train", "a") as f:
            for i in range(length):
                temp = []
                temp.append(line[i])
                temp.append(labels[i])
                l = " ".join(temp)
                f.write(l.encode("utf8")+"\n")
            f.write("\n")


def test():
    with open("F:/guomianzhuang/concept/cpt.train") as f:
        lines = f.readlines()
        count = 0
        count2 = 0
        for line in lines:
            line = line.replace("\n", "").decode("utf8")
            if u"。" in line:
                count+=1
            if line == "":
                count2+=1
                print "test"
                continue
            ls = line.split(" ")
            if len(ls) != 2:
                print len(ls)
                print ls
        print count, count2


def re():
    with open("F:/guomianzhuang/concept/train_select.txt") as f:
        lines = f.readlines()
        count = 0
        for line in lines:
            count += 1
            if line.count("。") > 1:
                print count


if __name__ == '__main__':
    # generate_cpt_corpus()
    # print list("中华通".decode("utf8"))
    # print load_cpts()
    # sta_cpt()
    # generate_cpt_corpus(10)
    # print u"中国概念股".index(u"概念")
    # label_train()
    cpt_label_train()
    test()
    # re()