from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import jieba
import numpy as np
import logging
import os
import sys
import re
import multiprocessing


def read_corpus(fname):
    # with smart_open.smart_open(fname, encoding="utf-8") as f:
    #     for i, line in enumerate(f):
    #         if tokens_only:
    #             yield gensim.utils.simple_preprocess(line)
    #         else:
    #             # For training data, add tags
    #             yield TaggedDocument(gensim.utils.simple_preprocess(line), [i])
    # with open(fname, 'r', encoding='utf-8') as f:
    #     content = re.sub('，', '', f.read())
    #     sens = content.split('。')
    #     for i in range(len(sens)):
    #         cur_sen = sens[i].strip('\n')
    #         each_cut = list(jieba.cut(cur_sen))
    #         # yield TaggedDocument(gensim.utils.simple_preprocess(content[i]), [i])
    #         yield TaggedDocument(each_cut, [i])
    stopwords = list()
    documents = list()
    with open('../data/patent_abstract/stopwords_new.txt', 'r', encoding='utf-8') as stopfile:
        for stopline in stopfile.readlines():
            stopwords.append(stopline.strip())
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    tag = 0
    # for line in lines:
    #     content = re.sub('，', '', line)
    #     sent_list = content.split('。')
    #     l = len(sent_list)
    #     if '\n' in sent_list[l-1]:
    #         sent_list.pop(l-1)
    #     for sen in sent_list:
    #         each_cut = list(jieba.cut(sen))
    #         yield TaggedDocument(each_cut, [tag])
    #         tag += 1
    for line in lines:
        line_split = line.split(' ::  ')
        if len(line_split) == 2:
            print('处理第%d个专利摘要......' % (tag + 1))
            # content = re.sub('[，。；、]+', '', line_split[1])
            # content = content.strip()
            content =line_split[1].strip()
            each_cut = list(jieba.cut(content))
            word_list = [word for word in each_cut if word not in stopwords]
            # i = 0
            # while i < len(each_cut):
            #     if each_cut[i] in stopwords:
            #         each_cut.pop(i)
            #     else:
            #         i += 1
            # print(each_cut)
            documents.append(TaggedDocument(word_list, [tag]))
            # yield TaggedDocument(each_cut, [tag])
            tag += 1
    return documents


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp1 = sys.argv[1:3]
    train_file = inp
    train_corpus = read_corpus(train_file)
    print(len(train_corpus))
    dim = 100    # 句向量的维度
    model = Doc2Vec(vector_size=dim, window=5, min_count=1, dm=1, epochs=10, workers=multiprocessing.cpu_count())
    model.build_vocab(train_corpus)
    # model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    # model = Doc2Vec(train_corpus, vector_size=200, window=2, min_count=1, dm=1, workers=multiprocessing.cpu_count())
    model.save(outp1)

    # vector_dict = model.docvecs
    # # print(len(vector_dict))
    # vectors = np.zeros((1, dim))
    # for num in range(0, len(vector_dict)):
    #     if num == 0:
    #         vectors[num] = vector_dict[num].reshape(1, dim)
    #     else:
    #         row = vector_dict[num].reshape(1, dim)
    #         vectors = np.row_stack((vectors, row))
    # np.save(outp2, vectors)
    #
    # print(vectors.shape)

    # python train_sec2vec_ZH.py D:\PycharmProjects\Dataset\keywordEX\patent\_all_abstract.txt ..\data\model\sen2vec\patent\all_100_dm_10.model ..\data\model\sen2vec\patent\all_100_dm_10.npy
    # python3 train_sec2vec_ZH.py ../data/patent_abstract/_bxk_abstract.txt ../data/model/sen2vec/patent/bxk_50_dm_20.model ../data/model/sen2vec/patent/bxk_50_dm_20.npy
    # python train_sec2vec_ZH.py..\data\patent_abstract / _bxk_abstract.txt..\data\model\sen2vec\patent\bxk_100_dm_10.model..\data\model\sen2vec\patent\bxk_100_dm_10.vector
# lee_train_file = '../data/raw/SemEval2010_train_raw.txt'
# lee_test_file = '../data/SemEval2010/train/C-41.txt.final'
#
#
#
# train_corpus = list(read_corpus(lee_train_file))
# test_corpus = list(read_corpus(lee_test_file, tokens_only=True))
# model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
# model.build_vocab(train_corpus)
# model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
# model.save(outp1)
# model.wv.save_word2vec_format(outp2, binary=False)
# print(model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires']))
