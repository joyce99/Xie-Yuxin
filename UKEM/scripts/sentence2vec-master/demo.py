#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""

"""
import multiprocessing
import logging
import sys
import os
from word2vec import Word2Vec, Sent2Vec, LineSentence

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

# input_file = r'D:\PycharmProjects\Dataset\keywordEX\patent\all_fc_rm_abstract_NEW.txt'
# model = Word2Vec(LineSentence(input_file), size=200, window=5, sg=1, hs=1, min_count=1, workers=multiprocessing.cpu_count())
# model.save(r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_abstract_200_mincount1.model')
# model.save_word2vec_format(r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_abstract_200_mincount1.vec')

sent_file = r'D:\PycharmProjects\Dataset\keywordEX\patent\kTVq\kTVq_fc_rm_abstract.txt'
model = Sent2Vec(LineSentence(sent_file), model_file=r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_abstract_100_mincount1.model')
model.save_sent2vec_format(r'D:\PycharmProjects\Dataset\keywordEX\patent\sent2vec\kTVq_fc_rm_abstract_100.vec')

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)