#!/usr/bin/env python
# encoding: utf-8
"""
    @time: 12/27/2017 2:41 PM
    @desc:
        
    @author: guomianzhuang
"""
from shutil import copy
import os
import re
dest_dir = "C:/Users/guomianzhuang/PycharmProjects/AlgorithmZhuang/"
source_dir = "C:/Users/guomianzhuang/PycharmProjects/MyAlgorithm/"

if not dest_dir.endswith('/'):
    dest_dir += '/'
if not source_dir.endswith('/'):
    source_dir += '/'
if os.path.isdir(dest_dir) and os.path.isdir(source_dir):
    for root, dirs, files in os.walk(source_dir):
        for i in xrange (0, files.__len__()):
            sf = os.path.join(root, files[i])
            dst = re.sub('([A-Za-z]:/.*?)/', dest_dir, root)
            if not os.path.exists(dst):
                os.makedirs(dst)
            copy(sf, dst)
    print 'Done!'

else:
    raise Exception('Wrong path entered!')

