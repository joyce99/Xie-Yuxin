import numpy as np
import operator
import jieba
import re
import logging
import os
import sys
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from embeddings import read, plot_with_labels
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

class patent_ZH:
    def __init__(self, content, doc_num):
        self.label = -1
        self.doc_num = doc_num
        self.docvec = None
        self.content = content

def get_class_num(labels):
    class_num = {}
    for label in labels:
        if label not in class_num:
            class_num[label] = 1
        else:
            class_num[label] += 1
    class_num = dict(sorted(class_num.items(), key=operator.itemgetter(0)))
    return class_num

def get_class_title(labels):
    class_title = {}
    for i, label in enumerate(labels):
        if label not in class_title:
            class_title[label] = [i]
        else:
            class_title[label].append(i)
    class_title = dict(sorted(class_title.items(), key=operator.itemgetter(0)))
    return class_title

def get_DBSCAN_clusters(vectors,labels):    # 根据DBSCAN聚类后的标签labels整理各类的向量，存放在字典clusters
    clusters = {}
    for i in range(len(labels)):
        if labels[i] not in clusters:
            clusters[labels[i]] = vectors[i]
        elif labels[i] in clusters:
            cur_vec = vectors[i]
            cur_cluster = clusters[labels[i]]
            clusters[labels[i]] = np.row_stack((cur_cluster, cur_vec))
    return clusters

def get_centers(model, clusters, method):  # 获得各个类的中心点(噪音类除外)
    centers = {}
    if method == 'DBSCAN':
        for label in clusters:
            if label == -1:     #如果是噪音类
                continue
            else:
                cur_vectors = clusters[label]
                cur_center = np.mean(cur_vectors, 0)
                centers[label] = cur_center
    elif method == 'Kmeans':
        label = 0
        for center in model.cluster_centers_:
            centers[label] = center
            label += 1
    return centers


def get_distance(cur_vector, cur_center, method):   # 获得与中心点的距离(余弦相似度 or 欧式距离)
    if method == 'cos':
        num = float(np.dot(cur_vector, cur_center.T))
        vec_norm = np.linalg.norm(cur_vector) * np.linalg.norm(cur_center)
        cos = num / vec_norm
        sim = 0.5 + 0.5 * cos   # 归一化
        return sim
    elif method == 'ED':
        distance = np.linalg.norm(cur_vector - cur_center)
        return distance

def distance_sort(ind2vec, cur_center, method):     # 获得根据与中心点距离大小排序后的{词向量：与中心点的距离}
    index_distance = {}
    for index in ind2vec:
        distance = get_distance(ind2vec[index], cur_center, method)
        index_distance[index] = distance
    if method == 'cos':
        sorted_distance = sorted(index_distance.items(), key=operator.itemgetter(1), reverse=True)
    else:
        sorted_distance = sorted(index_distance.items(), key=operator.itemgetter(1))
    sorted_index_distance = dict(sorted_distance)
    return sorted_index_distance

def get_index2vectors(word2ind, wordvecs, filename=None, cur_str=None):    # 获得测试文本中所有词的词向量
    ind2vec = {}
    if filename:
        test_file = open(filename, 'r', encoding='utf-8')
        for line in test_file.readlines():
            curline_words = line.split(' ')
            for word in curline_words:
                if word == '\n':
                    continue
                elif word in word2ind:
                    cur_index = word2ind[word]
                    cur_vec = wordvecs[cur_index]
                    ind2vec[cur_index] = cur_vec
        test_file.close()
    elif cur_str:
        rm_str = re.sub("[\s+\.\!\/_,;\[\]><•¿#&«»∗`{}=|1234567890¡?():$%^*(+\"\']+|[+！，。？；：、【】《》“”‘’~@#￥%……&*（）''""]+", " ", cur_str)
        seg_list = jieba.cut(rm_str)
        for word in seg_list:
            if word == '\n':
                continue
            elif word in word2ind:
                cur_index = word2ind[word]
                cur_vec = wordvecs[cur_index]
                ind2vec[cur_index] = cur_vec
    return ind2vec


def get_most_label(ind2vec, clusters, dim):     # 获得测试文本中单词数最多的类别
    class_vector = {}
    for index in ind2vec:
        for label in clusters:
            if ind2vec[index] in clusters[label]:
                if label not in class_vector:
                    class_vector[label] = ind2vec[index].reshape(1, dim)
                else:
                    class_vector[label] = np.row_stack((class_vector[label], ind2vec[index]))
                break
    assert len(class_vector) > 0
    class_vector = dict(sorted(class_vector.items(), key=operator.itemgetter(0)))
    if class_vector.keys() == [-1]:
        most_label = -1
        print('所有词向量均为噪音！')
        return most_label
    else:
        most_label = 0
        most_num = class_vector[most_label].shape[0]
    for label in class_vector:
        if label == -1 or label == 0:
            continue
        else:
            if class_vector[label].shape[0] > most_num:
                most_num = class_vector[label].shape[0]
                most_label = label
    print('本文中%d类包含的单词最多，单词数为：%d,占本文单词的%f%%' % (most_label, most_num, most_num * 100.0 / len(ind2vec)))
    return most_label

def mainZH():
    # model = Word2Vec.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_100_SG.model')
    patent_file = open('../data/patent_abstract/_bxk_abstract.txt', 'r', encoding='utf-8')
    # dim = 100
    # num = 0
    # docvecs = np.zeros((1, dim))
    # for line in patent_file.readlines():
    #     content = re.sub('[，。；、]+', '', line)
    #     content = content.strip()
    #     each_cut = list(jieba.cut(content))
    #     print('处理第%d个专利摘要......' % (num + 1))
    #     num += 1
    #     wordvecs = np.zeros((1, dim))
    #     for word in each_cut:
    #         try:
    #             cur_wordvec = model.wv[word].reshape(1, dim)
    #             wordvecs = np.concatenate((wordvecs, cur_wordvec), axis=0)
    #         except Exception as e:
    #             pass
    #     wordvecs = np.delete(wordvecs, 0, 0)
    #     # print('wordvecs.shape=' + str(wordvecs.shape))
    #     cur_docvec = np.mean(wordvecs, axis=0).reshape(1, 100)
    #     # print('cur_docvec.shape=' + str(cur_docvec.shape))
    #     docvecs = np.concatenate((docvecs, cur_docvec), axis=0)
    # docvecs = np.delete(docvecs, 0, 0)
    # np.save('../data/model/word2vec/patent/bxk_all_100_SG.npy', docvecs)
    docvecs = np.load('../data/model/word2vec/patent/bxk_all_100_SG.npy')
    # disMat = sch.distance.pdist(docvecs, 'cosine')
    # Z = sch.linkage(disMat, method='average')
    print('DBSCAN聚类中......')
    log_file = open('../data/all_word2vec_log.txt', 'a', encoding='utf-8')
    myeps = 0.2
    while myeps <= 0.9:
        for my_min_samples in range(3, 11):
            cluster = DBSCAN(eps=myeps, min_samples=my_min_samples).fit_predict(docvecs)
            class_num = get_class_num(cluster)
            print('eps=%f, min_samples=%d' % (myeps, my_min_samples))
            n_clusters_ = len(set(cluster)) - (1 if -1 in cluster else 0)
            print('聚类的类别数目(除噪音外)：%d' % (n_clusters_))
            ratio = len(cluster[cluster[:] == -1]) / len(cluster)
            print('噪音率:' + str(ratio))
            log_file.write('eps = %f ,min_samples = %d \n聚类的类别数目（除噪音外）：%d , 噪音率: %f\n' % (myeps, my_min_samples, n_clusters_, ratio))
            print('聚类结果为：')
            log_file.write('聚类结果为：\n')
            for label in class_num:
                print(str(label) + ':' + str(class_num[label]))
                log_file.write(str(label) + ':' + str(class_num[label]) + '\t;\t')
            print('----------------------------------------------------------------')
            log_file.write('\n------------------------------------------------------------------\n')
        myeps = myeps + 0.01
    # db_model = DBSCAN(eps=0.5, min_samples=3).fit(docvecs)
    # cluster = db_model.labels_
    # 将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
    # plt.figure(num='层次聚类结果', figsize=(8, 8))
    # P = sch.dendrogram(Z)
    # plt.savefig('bxk_all_word2vec.png')
    # 根据linkage matrix Z得到聚类结果:
    # cluster = sch.fcluster(Z, 0.22, 'distance', depth=2)

    # ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
    # cluster = ac.fit_predict(docvecs)
    # cluster = KMeans(n_clusters=3, random_state=9).fit_predict(docvecs)
    # file_list = get_label(file_list, cluster)

    # patent_list = get_label(patent_list, cluster)
    # my_result = get_result(file_list)
    # my_result = get_patent_result(patent_list)

    # labels_unique = np.unique(cluster)
    # n_clusters_ = len(labels_unique)
    # print('聚类的类别数目：%d' % n_clusters_)
    # class_num = get_class_num(cluster)
    # ratio = len(cluster[cluster[:] == -1]) / len(cluster)
    # print('认为是噪音的数据比例：%d' % ratio)
    # print('聚类结果为：')
    # print(class_num)
    # ratio = len(cluster[cluster[:] == -1]) / len(cluster)
    # print('噪音率:' + str(ratio))
    # log_file.write('eps = %f ,min_samples = %d \n聚类的类别数目（除噪音外）：%d , 噪音率: %f\n' % (myeps, my_min_samples, n_clusters_, ratio))

    # print('聚类结果为：')
    # log_file.write('聚类结果为：\n')
    # for label in class_num:
    #     print(str(label) + ':' + str(class_num[label]))
        # log_file.write(str(label) + ':' + str(class_num[label]) + '\t;\t')
    print('----------------------------------------------------------------')
    log_file.close()
    patent_file.close()
# def mainEN():
    # embedding_file = open(r'D:\PycharmProjects\Dataset\keywordEX\wikiZH_100_SG.vector', 'r', encoding='utf-8', errors='surrogateescape')
    # words, wordvecs = read(embedding_file, dtype=float)
    # assert len(words) == wordvecs.shape[0]
    # word2ind = {word: i for i, word in enumerate(words)}
    # print('DBSCAN聚类中......')
    # db_model = DBSCAN(eps=1, min_samples=5, algorithm='kd_tree', n_jobs=-1).fit(wordvecs)
    # # db_model = KMeans(n_clusters=4, max_iter=500, random_state=0).fit(wordvecs)
    # db_labels = db_model.labels_
    # n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    # print('聚类的类别数目(噪音类除外)：%d' % n_clusters)
    # ratio = len(db_labels[db_labels[:] == -1]) / len(db_labels)
    # print('噪音率:' + str(ratio))
    # clusters = get_DBSCAN_clusters(wordvecs, db_labels)
    # print('聚类结果为：')
    # for label in clusters:
    #     print(str(label) + ':' + str(clusters[label].shape[0]) )
    # centers = get_centers(db_model, clusters, 'DBSCAN')
    # # centers = get_centers(db_model, clusters, 'Kmeans')
    # abstract = '本发明公开一种具有语音交互功能的声控空调器，通过用户发出的语音指令信息直接对空调器进行控制，并在对空调进行语音控制过程中通过反馈语音指令信息给用户确认，实现用户与空调的语音交互。该技术方案能够完全摆脱遥控器实现对空调的控制，操作方便，同时，语音交互方式具有灵活性，能够满足不同用户个性化的要求，提高了用户的体验。'
    # ind2vec_test = get_index2vectors(word2ind, wordvecs, cur_str=abstract)
    # # ind2vec_test = get_index2vectors(word2ind, wordvecs,filename='../data/SemEval2010/train_removed/C-41.txt')
    # most_label = get_most_label(ind2vec_test, clusters, wordvecs.shape[1])
    # index_distance = distance_sort(ind2vec_test, centers[most_label], 'cos')
    # top_k = 0
    # for index in index_distance:
    #     cur_word = words[index]
    #     top_k += 1
    #     print('%d、%s' % (top_k, cur_word))
    #     print(index_distance[index])
    #     if top_k >= 30 or top_k >= len(index_distance):
    #         break
    # embedding_file.close()


if __name__ == '__main__':
    mainZH()

