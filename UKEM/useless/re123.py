import logging
import os
import sys
import re
import jieba
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import operator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import datasets
from embeddings import read
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.manifold import TSNE
from sklearn.cluster import Birch
from extractTrain import myfile

class file_EN:
    def __init__(self, name, doc_num):
        self.name = name
        self.label = -1
        self.doc_num = doc_num
        self.docvec = None

class patent_ZH:
    def __init__(self, content, doc_num):
        self.label = -1
        self.doc_num = doc_num
        self.docvec = None
        self.content = content
def get_label(file_list,cluster):
    f_num = 0
    for label in cluster:
        cur_file = file_list[f_num]
        cur_file.label = label
        f_num += 1
    return file_list

def get_result(file_list):
    my_dict = {}
    for f in file_list:
        if f.label not in my_dict:
            my_dict[f.label] = [f.name]
        else:
            my_dict[f.label].append(f.name)
    my_dict = dict(sorted(my_dict.items(), key=operator.itemgetter(0)))
    return my_dict

def get_patent_result(patent_list):
    my_dict = {}
    for patent in patent_list:
        if patent.label not in my_dict:
            my_dict[patent.label] = [patent.content]
        else:
            my_dict[patent.label].append(patent.content)
    my_dict = dict(sorted(my_dict.items(), key=operator.itemgetter(0)))
    return my_dict

def plot_with_labels(low_dim_embs, colors, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(50, 50))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y, c=colors[i])
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

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
# X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
#                                       noise=.05)
# X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#                random_state=9)
# X = np.concatenate((X1, X2))
# y_pred = [-1 for i in range(6000)]
# plt.scatter(X[:, 0], X[:, 1], marker='o',c=y_pred)
# plt.show()
# y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
# y_pred = DBSCAN(eps=0.1, min_samples=10).fit_predict(X)
# print(y_pred.shape)
# n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)
# print('聚类的类别数目：%d' % (n_clusters_))
# ratio = len(y_pred[y_pred[:] == -1]) / len(y_pred)
# print('认为是噪音的数据比例：%d' % (ratio))
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()


if __name__ == '__main__':
    # folder = r"../data/SemEval2010/mine"
    # filters = ['C', 'H', 'I', 'J']
    # truth = {'C': [], 'H': [], 'I': [], 'J': []}
    # num = 0
    # file_list = []
    # train_file = open('../data/SemEval2010/new_line_doc.txt', 'w', encoding='utf-8')
    # for name_start in filters:
    #     for i in range(100):
    #         cur_name = name_start + '-' + str(i) + '.txt.final'
    #         abs_name = os.path.join(folder, cur_name)
    #         isfile = os.path.isfile(abs_name)
    #         if isfile:
                # with open(abs_name, 'r', encoding='utf-8') as curf:
                # for line in curf.readlines():
                # train_file.write(re.sub('\n', ' ', line))
                # train_file.write('\n')
                # cur_file = file_EN(cur_name, num)
                # file_list.append(cur_file)
                # truth[name_start].append(num)
                # num += 1
    # train_file.close()
    # print(truth)

    ## DBSCAN聚类法
    # embedding_file = open(r'D:\PycharmProjects\Dataset\keywordEX\old\all_50_SG.vector', 'r', encoding='utf-8', errors='surrogateescape')
    # model = Doc2Vec.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10.model')
    # embedding_file = open(r'D:\PycharmProjects\KeywordExtraction\data\model\word2vec\patent\bxk_50_SG.vector', 'r', encoding='utf-8', errors='surrogateescape')
    # words, vectors = read(embedding_file, dtype=float)
    # plot_only = 5000
    # log_file = open('../data/allpatent_log.txt', 'a', encoding='utf-8')
    # myeps = 2
    # while myeps <= 2.5:
    #     for my_min_samples in range(5,7):
    #         print('DBSCAN聚类中......')
    #         db_labels = DBSCAN(eps=myeps, min_samples=my_min_samples, n_jobs=-1 ).fit_predict(vectors)
    #         # db_labels = DBSCAN(eps=myeps, min_samples=my_min_samples, algorithm='ball_tree').fit_predict(vectors)
    #         class_num = get_class_num(db_labels)
    #         print('eps=%f, min_samples=%d' % (myeps, my_min_samples))
    #         n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    #         print('聚类的类别数目(除噪音外)：%d' % (n_clusters_))
    #         ratio = len(db_labels[db_labels[:] == -1]) / len(db_labels)
    #         print('噪音率:' + str(ratio))
    #         log_file.write('eps = %f ,min_samples = %d \n聚类的类别数目（除噪音外）：%d , 噪音率: %f\n' % (myeps, my_min_samples, n_clusters_, ratio))
    #         print('聚类结果为：')
    #         log_file.write('聚类结果为：\n')
    #         for label in class_num:
    #             print(str(label) + ':' + str(class_num[label]))
    #             log_file.write(str(label) + ':' + str(class_num[label]) + '\t;\t')
    #         print('----------------------------------------------------------------')
    #         log_file.write('\n------------------------------------------------------------------\n')
    #     myeps = myeps + 0.1

    dim = 100
    model = Doc2Vec.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10_5.model')
    patent_list = []
    docvecs = np.zeros((1,dim))
    num = 0
    with open('../data/patent_abstract/_bxk_abstract.txt', 'r', encoding='utf-8') as curf:
        for line in curf.readlines():
            content = re.sub('[，。；、]+', '', line)
            content = content.strip()
            each_cut = list(jieba.cut(content))
            line = line.strip()
            cur_patent = patent_ZH(line, num)
            cur_docvec = model.infer_vector(each_cut)
            cur_patent.docvec = cur_docvec
            print('读取第%d个专利摘要......' % (num + 1))
            if num == 0:
                docvecs[0] = cur_docvec.reshape(1,dim)
            else:
                docvecs = np.row_stack((docvecs, cur_docvec.reshape(1, dim)))
            patent_list.append(cur_patent)
            num += 1
    print(docvecs.shape)
    # np.save('../data/model/sen2vec/patent/bxk_all_100_dm_10_5.npy', docvecs)
    # 1. 层次聚类
    # 生成点与点之间的距离矩阵,这里用的欧氏距离:
    # sentvecs = np.load('../data/model/sen2vec/patent/bxkdoc_100_dm_40_5.npy')

    # docvecs = np.load('../data/model/sen2vec/patent/bxk_all_100_dm_10_5.npy')
    # log_file = open('../data/all_Doc2vec_log.txt', 'a', encoding='utf-8')
    # myeps = 2.5
    # while myeps <= 4:
    #     for my_min_samples in range(3, 8):
    #         cluster = DBSCAN(eps=myeps, min_samples=my_min_samples, n_jobs=-1).fit_predict(docvecs)
            # sentvecs = np.load(r'D:\PycharmProjects\KeywordExtraction\data\model\sen2vec\SE2010\new_SEdoc_50_dm_40.vector.npy')
    # disMat = sch.distance.pdist(docvecs, 'cosine')
    # Z = sch.linkage(disMat, method='average')
    # 将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
    # plt.figure(num='层次聚类结果', figsize=(8, 8))
    # P=sch.dendrogram(Z)
    # plt.savefig('bxk_all_100_10_5.png')
    # 根据linkage matrix Z得到聚类结果:

    cluster = KMeans(n_clusters=3, random_state=9).fit_predict(docvecs)

    # ac = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
    # cluster = ac.fit_predict(docvecs)
    # cluster = KMeans(n_clusters=3, random_state=9).fit_predict(docvecs)
    # file_list = get_label(file_list, cluster)

    patent_list = get_label(patent_list, cluster)
    # my_result = get_result(file_list)
    my_result = get_patent_result(patent_list)
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    # print('聚类结果为：')
    # print(class_num)
    # ratio = len(cluster[cluster[:] == -1]) / len(cluster)
    # print('噪音率:' + str(ratio))
    # log_file.write('eps = %f ,min_samples = %d \n聚类的类别数目（除噪音外）：%d , 噪音率: %f\n' % (myeps, my_min_samples, n_clusters_, ratio))
    print('聚类结果为：')
    # log_file.write('聚类结果为：\n')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
        # log_file.write(str(label) + ':' + str(class_num[label]) + '\t;\t')
    # print('----------------------------------------------------------------')
            # log_file.write('\n------------------------------------------------------------------\n')
        # myeps = myeps + 0.1
    # log_file.close()
    # class_title = get_class_title(cluster)
    with open('../data/patent_abstract/bxk_all_100_dm_10_5_KMeans.txt', 'w', encoding='utf-8') as result_f:
        for label in my_result:
            result_f.write(str(label) + ':' +'\n')
            for patent in my_result[label]:
                result_f.write(patent + ' ;' + '\n')

    # print(class_title)
    # truth = {3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
    #          0: [59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122],
    #          2: [123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182],
    #          1: [183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243]}
    # correct = 0
    # print('truth:')
    # print(truth)
    # for label_train in truth:
    #     for label_test in class_title:
    #         if label_test == label_train:
    #             for j in class_title[label_test]:
    #                 if j in truth[label_train]:
    #                     correct += 1
    #             break
    # print('聚类正确率：%f%%' % (correct/244.0*100))
    # print('----------------------------------------------------------------')
