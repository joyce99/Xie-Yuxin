import re
import jieba
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import operator
from sklearn.cluster import KMeans
from embeddings import read
from sklearn import metrics
from textrank4zh import TextRank4Keyword

class patent_ZH:
    def __init__(self, content, doc_num, ipc):
        self.label = -1
        self.content = content
        self.doc_num = doc_num
        self.docvec = None
        self.ipc = ipc

def get_label(file_list,cluster):
    f_num = 0
    for label in cluster:
        cur_file = file_list[f_num]
        cur_file.label = label
        f_num += 1
    return file_list

def get_patent_result(patent_list):
    result_dict = dict()
    for patent in patent_list:
        if patent.label not in result_dict:
            result_dict[patent.label] = [patent.content]
        else:
            result_dict[patent.label].append(patent.content)
    result_dict = dict(sorted(result_dict.items(), key=operator.itemgetter(0)))
    return result_dict

def get_patent_ipc(patent_list):
    ipc_dict = dict()
    for patent in patent_list:
        if patent.label not in ipc_dict:
            ipc_dict[patent.label] = [patent.ipc]
        else:
            ipc_dict[patent.label].append(patent.ipc)
    ipc_dict = dict(sorted(ipc_dict.items(), key=operator.itemgetter(0)))
    return ipc_dict

def get_class_num(labels):
    class_num = dict()
    for label in labels:
        if label not in class_num:
            class_num[label] = 1
        else:
            class_num[label] += 1
    class_num = dict(sorted(class_num.items(), key=operator.itemgetter(0)))
    return class_num

def get_stopwords(fname):
    stop_file = open(fname, 'r', encoding='utf-8')
    stopwords = list()
    for line in stop_file.readlines():
        stopwords.append(line.strip())
    return stopwords

def get_centers(predefined_dict, dim=100):  # 获得各个类的中心点(噪音类除外)
    centers = np.zeros((len(predefined_dict), dim))
    for label in predefined_dict:
        if label == -1:  # 如果是噪音类
            continue
        else:
            cur_vectors = predefined_dict[label]
            kmeans_model = KMeans(n_clusters=1, init='k-means++', max_iter=10000).fit(cur_vectors)
            cur_center = kmeans_model.cluster_centers_[0].reshape(1, dim)
            # cur_center = np.mean(cur_vectors, axis=0).reshape(1, dim)
            # print(cur_vectors.shape)
            # print(np.mean(cur_vectors, axis=0).shape)
            centers[label] = cur_center
    return centers

def kmeans1():      # Doc2vec
    dim = 100
    model = Doc2Vec.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_abstract_100_nostop.model')
    patent_list = list()
    docvecs = np.zeros((1, dim))
    num = 0
    stopfile = open('../data/patent_abstract/stopwords_new.txt', 'r', encoding='utf-8')
    stopwords = list()
    for line in stopfile.readlines():
        stopwords.append(line.strip())
    # with open('D:\PycharmProjects\Dataset\keywordEX\patent\_all_label_abstract.txt', 'r', encoding='utf-8') as curf:
    #     for line in curf.readlines():
    #         content = re.sub('[，。；、]+', '', line)
    #         content = content.strip()
    #         each_cut = list(jieba.cut(content))
    #         line = line.strip()
    #         cur_patent = patent_ZH(line, num)
    #         cur_docvec = model.infer_vector(each_cut)
    #         cur_patent.docvec = cur_docvec
    #         print('读取第%d个专利摘要......' % (num + 1))
    #         if num == 0:
    #             docvecs[0] = cur_docvec.reshape(1, dim)
    #         else:
    #             docvecs = np.row_stack((docvecs, cur_docvec.reshape(1, dim)))
    #         patent_list.append(cur_patent)
    #         num += 1
    with open('D:\PycharmProjects\Dataset\keywordEX\patent\_bxk_label_abstract.txt', 'r', encoding='utf-8') as curf:
        for line in curf.readlines():
            line_split = line.split(' ::  ')
            if len(line_split) == 2:
                content_rm = line_split[1].strip()
                line_cut = list(jieba.cut(content_rm))
                line_words = [word for word in line_cut if word not in stopwords]
                content = line_split[1].strip()
                cur_patent = patent_ZH(content, num, line_split[0])
                cur_docvec = model.infer_vector(line_words)
                cur_patent.docvec = cur_docvec
                print('读取第%d个专利摘要......' % (num + 1))
                if num == 0:
                    docvecs[0] = cur_docvec.reshape(1, dim)
                else:
                    docvecs = np.row_stack((docvecs, cur_docvec.reshape(1, dim)))
                patent_list.append(cur_patent)
                num += 1
    print(docvecs.shape)
    cluster = KMeans(n_clusters=3, init='k-means++', max_iter=10000).fit_predict(docvecs)
    patent_list = get_label(patent_list, cluster)
    my_ipc = get_patent_ipc(patent_list)
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    print('聚类结果为：')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
    # with open('../data/patent_abstract/cengci/bxk_all_100_10_5_cengci.txt', 'w', encoding='utf-8') as result_f:
    with open('../data/patent_abstract/Kmeans/bxk_abstract_nostop_doc2vecTest_100.txt', 'w',
              encoding='utf-8') as result_f:
        result_f.write('聚类结果为：\n')
        for label in class_num:
            result_f.write(str(label) + ':' + str(class_num[label]) + '\n')
        for label in my_ipc:
            result_f.write('类标签为:' + str(label) + ':' + '\n')
            result_f.write(str(class_num[label]) + '条专利' + '\n')
            for ipc in my_ipc[label]:
                result_f.write(str(label) + ':  ' + ipc + '\n')
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(docvecs, cluster))
    stopfile.close()

def kmeans2():      # sent2vec
    embedding_file = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\sent2vec\bxd_fc_rm_techField.vec', 'r',
                          encoding='utf-8', errors='surrogateescape')
    sent_num, sentvecs = read(embedding_file, dtype=float)
    patent_list = list()
    num = 0
    with open('D:\PycharmProjects\Dataset\keywordEX\patent\_bxd_label_techField.txt', 'r', encoding='utf-8') as curf:
        for line in curf.readlines():
            line_split = line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                cur_patent = patent_ZH(content, num, line_split[0])
                # ipc_list.append(line_split[0])
                print('读取第%d个专利摘要......' % (num + 1))
                patent_list.append(cur_patent)
                num += 1
    print(sentvecs.shape)
    cluster = KMeans(n_clusters=3, init='k-means++', max_iter=10000).fit_predict(sentvecs)
    patent_list = get_label(patent_list, cluster)
    my_ipc = get_patent_ipc(patent_list)
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    print('聚类结果为：')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
    # with open('../data/patent_abstract/cengci/bxk_all_100_10_5_cengci.txt', 'w', encoding='utf-8') as result_f:
    with open('../data/patent_abstract/Kmeans/bxd_techField_sent2vec_Test.txt', 'w', encoding='utf-8') as result_f:
        result_f.write('聚类结果为：\n')
        for label in class_num:
            result_f.write(str(label) + ':' + str(class_num[label]) + '\n')
        for label in my_ipc:
            result_f.write('类标签为:' + str(label) + ':' + '\n')
            result_f.write(str(class_num[label]) + '条专利' + '\n')
            for ipc in my_ipc[label]:
                result_f.write(str(label) + ':  ' + ipc + '\n')
    embedding_file.close()


def keyword_extraction(log_file_name, test_name, wordvec_name, birch_model, centers, dim=100, topn=20):
    log_file = open(log_file_name, 'w', encoding='utf-8')
    wordvec_file = open(wordvec_name, 'r', encoding='utf-8', errors='surrogateescape')
    stopwords = get_stopwords('../data/patent_abstract/stopwords_new.txt')
    keywordstop = get_stopwords('../data/patent_abstract/keywordstop.txt')
    words, wordvecs = read(wordvec_file, dtype=float)
    word2ind = {word: i for i, word in enumerate(words)}
    with open(test_name, 'r', encoding='utf-8') as test_file:
        num = 0
        for test_line in test_file.readlines():
            line_split = test_line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                print('第%d条专利摘要：' % (num+1))
                print(content)
                log_file.write('第%d条专利摘要：\t\t%s\n' % (num+1, line_split[0]))
                log_file.write('%s\n' % content)
                test_line_words = list(jieba.cut(content))
                line_words = list()
                line_vecs = list()
                for word in test_line_words:
                    if word not in stopwords and word not in keywordstop and word in word2ind and len(word)>1:
                        line_words.append(word)
                        cur_wordvec = wordvecs[word2ind[word]].reshape(1, dim)
                        line_vecs.append(cur_wordvec)
                assert len(line_words) == len(line_vecs)
                ind2vec = get_index2vectors(word2ind, wordvecs, line_words)
                most_label = get_most_label(line_vecs, birch_model)
                center = centers[most_label]
                sorted_index_distance = distance_sort(ind2vec, center, 'cos')
                keyword_num = 0
                # print('-------------keywords-----------------')
                # log_file.write('-------------keywords-----------------\n')
                # for our_item in list(sorted_index_distance.items()):
                #     cur_word = words[our_item[0]]
                #     cur_dis = our_item[1]
                #     log_file.write('%s\t\t%f\n' % (cur_word, cur_dis))
                #     print(cur_word + '\t' + str(cur_dis))
                #     keyword_num += 1
                #     if keyword_num >= topn:
                #         break
                tr4w = TextRank4Keyword(stop_words_file = '../data/patent_abstract/TextRankstop.txt')
                tr4w.analyze(text=content, lower=True, window=3, vertex_source = 'words_no_stop_words', pagerank_config={'alpha': 0.85})
                print('textrank-----------ours-----------------')
                log_file.write('textrank----ours-----------------\n')
                for textrank_item, our_item in zip(tr4w.get_keywords(20, word_min_len=2), list(sorted_index_distance.items())):
                    cur_word = words[our_item[0]]
                    cur_dis = our_item[1]
                    log_file.write('%s\t\t\t' % textrank_item.word)
                    log_file.write('%s\n' % cur_word)
                    print(textrank_item.word + '%f' % textrank_item.weight + '\t\t' + cur_word + '%f' % cur_dis)
                    keyword_num += 1
                    if keyword_num >= topn:
                        break
                print('-----------------------------------------------------------------')
                log_file.write('------------------------------------------------------------------\n')
                num += 1
    wordvec_file.close()
    log_file.close()


if __name__ == '__main__':
    # sent2vec_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\sent2vec\bxd_fc_rm_techField_100.vec'
    embedding_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_abstract_100_mincount1.vec'
    birch_train_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\kTVq\_kTVq_label_techField.txt'
    cluster_result_name = '../data/patent_abstract/Birch/kTVq_techField_wordAVG_keywordTest_1.009_50.txt'
    log_file_name = r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\test\kTVq_textRankVSours_techField_wordAVG_1.009_50.txt'
    test_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\kTVq\_kTVq_label_abstract.txt'
    wordvec_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_abstract_100_mincount1.vec'
    # birch1()
    # birch_model, centers = birch2(sent2vec_name, birch_train_name, cluster_result_name)
    birch_model, centers = birch3(embedding_name, birch_train_name, cluster_result_name)
    keyword_extraction(log_file_name, test_name, wordvec_name, birch_model, centers)

