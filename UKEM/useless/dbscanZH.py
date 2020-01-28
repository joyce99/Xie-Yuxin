import re
import jieba
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
import operator
from embeddings import read
from sklearn.cluster import Birch
from sklearn import metrics


class patent_ZH:
    def __init__(self, content, doc_num, ipc):
        self.label = -1
        self.content = content
        self.doc_num = doc_num
        self.docvec = None
        self.ipc = ipc

def get_DBSCAN_clusters(vectors,labels):    # 根据Birch聚类后的标签labels整理各类的向量，存放在字典clusters
    clusters = dict()
    for i in range(len(labels)):
        if labels[i] not in clusters:
            clusters[labels[i]] = vectors[i]
        elif labels[i] in clusters:
            cur_vec = vectors[i]
            cur_cluster = clusters[labels[i]]
            clusters[labels[i]] = np.row_stack((cur_cluster, cur_vec))
    clusters = dict(sorted(clusters.items(), key=operator.itemgetter(0)))
    return clusters

def get_centers(clusters, dim=100):  # 获得各个类的中心点(噪音类除外)
    centers = np.zeros((len(clusters), dim))
    for label in clusters:
        if label == -1:  # 如果是噪音类
            continue
        else:
            cur_vectors = clusters[label]
            cur_center = np.mean(cur_vectors, axis=0).reshape(1, dim)
            centers[label] = cur_center
    return centers

def get_label(patent_list,cluster):
    f_num = 0
    for label in cluster:
        cur_file = patent_list[f_num]
        cur_file.label = label
        f_num += 1
    return patent_list

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

def get_index2vectors(word2ind, wordvecs, line_words):    # 获得测试文本中所有词的词向量
    ind2vec = dict()
    for word in line_words:
        cur_index = word2ind[word]
        cur_vec = wordvecs[cur_index]
        ind2vec[cur_index] = cur_vec
    return ind2vec

def get_distance(cur_vector, cur_center, method):   # 获得与中心点的距离(余弦相似度 or 欧式距离)
    if method == 'cos':
        num = float(np.dot(cur_vector, cur_center.T))
        vec_norm = np.linalg.norm(cur_vector) * np.linalg.norm(cur_center)
        cos = num / vec_norm
        sim = 0.5 + 0.5 * cos   # 归一化
        return sim
    elif method == 'ED':
        dist = np.linalg.norm(cur_vector - cur_center)
        return dist

def distance_sort(ind2vec, cur_center, method):     # 获得根据与中心点距离大小排序后的{词向量序号：与中心点的距离}
    index_distance = dict()
    for index in ind2vec:
        distance = get_distance(ind2vec[index], cur_center, method)
        index_distance[index] = distance
    if method == 'cos':
        sorted_distance = sorted(index_distance.items(), key=operator.itemgetter(1), reverse=True)
    else:
        sorted_distance = sorted(index_distance.items(), key=operator.itemgetter(1))
    sorted_index_distance = dict(sorted_distance)
    return sorted_index_distance

def get_stopwords():
    stop_file = open('../data/patent_abstract/stopwords_new.txt', 'r', encoding='utf-8')
    stopwords = list()
    for line in stop_file.readlines():
        stopwords.append(line.strip())
    return stopwords

def write_cluster_result(fname, class_num, my_ipc):
    with open(fname, 'w', encoding='utf-8') as result_f:
        result_f.write('聚类结果为：\n')
        for label in class_num:
            result_f.write(str(label) + ':' + str(class_num[label]) + '\n')
        for label in my_ipc:
            result_f.write('类标签为:' + str(label) + ':' + '\n')
            result_f.write(str(class_num[label]) + '条专利' + '\n')
            for ipc in my_ipc[label]:
                result_f.write(str(label) + ':  ' + ipc + '\n')

def get_most_label(line_vecs, birch_model):
    label_num = dict()
    for vec in line_vecs:
        cur_label = birch_model.predict(vec)
        if cur_label[0] not in label_num:
            label_num[cur_label[0]] = 1
        else:
            label_num[cur_label[0]] += 1
    label_num = dict(sorted(label_num.items(), key=operator.itemgetter(1), reverse=True))
    most_label = list(label_num.items())[0][0]
    return most_label

def dbscan1(model_name, dbscan_train_name, cluster_result_name):       # Doc2vec
    dim = 100
    model = Doc2Vec.load(model_name)
    patent_list = list()
    docvecs = np.zeros((1, dim))
    num = 0
    stopwords = get_stopwords()
    with open(dbscan_train_name, 'r', encoding='utf-8') as curf:
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
    # model = Birch(n_clusters=3, threshold=0.5, branching_factor=50).fit(docvecs)
    model = DBSCAN(eps=1.79, min_samples=4).fit(docvecs)
    cluster = model.labels_
    patent_list = get_label(patent_list, cluster)
    my_ipc = get_patent_ipc(patent_list)
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    print('聚类结果为：')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
    write_cluster_result(cluster_result_name, class_num, my_ipc)
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(docvecs, cluster))
    return model

def dbscan2(embedding_name, dbscan_train_name, cluster_result_name):       # sent2vec
    embedding_file = open(embedding_name, 'r', encoding='utf-8', errors='surrogateescape')
    sent_num, sentvecs = read(embedding_file, dtype=float)
    patent_list = list()
    num = 0
    dim = 100
    with open(dbscan_train_name, 'r', encoding='utf-8') as curf:
        for line in curf.readlines():
            line_split = line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                cur_patent = patent_ZH(content, num, line_split[0])
                cur_patent.docvec = sentvecs[num].reshape(1, dim)
                patent_list.append(cur_patent)
                print('读取第%d个专利摘要......' % (num + 1))
                num += 1
    print(sentvecs.shape)
    # model = Birch(threshold=0.5, branching_factor=50).fit(sentvecs)
    model = DBSCAN(eps=1.79, min_samples=4).fit(sentvecs)
    cluster = model.labels_
    patent_list = get_label(patent_list, cluster)
    my_ipc = get_patent_ipc(patent_list)
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    print('聚类结果为：')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
    write_cluster_result(cluster_result_name, class_num, my_ipc)
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(sentvecs, cluster))
    embedding_file.close()
    label_vecs = get_DBSCAN_clusters(sentvecs, cluster)
    centers = get_centers(label_vecs)
    return model, centers

def dbscan3(embedding_name, dbscan_train_name, cluster_result_name):       # 词向量加和平均
    embedding_file = open(embedding_name, 'r', encoding='utf-8', errors='surrogateescape')
    patent_list = list()
    dim = 100
    stopwords = get_stopwords()
    words, wordvecs = read(embedding_file, dtype=float)
    word2ind = {word: i for i, word in enumerate(words)}
    test_vecs = np.zeros((1, dim))
    with open(dbscan_train_name, 'r', encoding='utf-8') as test_file:
        num = 0
        for test_line in test_file.readlines():
            line_split = test_line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                cur_patent = patent_ZH(content, num, line_split[0])
                test_line_words = list(jieba.cut(content))
                line_words = [word for word in test_line_words if word not in stopwords]
                line_wordvecs = np.zeros((1, dim))
                for i in range(len(line_words)):
                    if line_words[i] in word2ind:
                        cur_wordindex = word2ind[line_words[i]]
                        cur_wordvec = wordvecs[cur_wordindex].reshape(1, dim)
                        if i == 0:
                            line_wordvecs[0] = cur_wordvec
                        else:
                            line_wordvecs = np.row_stack((line_wordvecs, cur_wordvec))
                cur_linevec = np.mean(line_wordvecs, axis=0).reshape(1, dim)
                cur_patent.docvec = cur_linevec
                patent_list.append(cur_patent)
                test_vecs = np.row_stack((test_vecs, cur_linevec))
                print('处理第%d条专利......' % (num+1))
            num += 1
        test_vecs = np.delete(test_vecs, 0 , 0)
    print(test_vecs.shape)
    # model = Birch(threshold=0.7, branching_factor=50).fit(test_vecs)
    model = DBSCAN(eps=0.7, min_samples=4).fit(test_vecs)
    cluster = model.labels_
    patent_list = get_label(patent_list, cluster)
    my_ipc = get_patent_ipc(patent_list)
    labels_unique = np.unique(cluster)
    n_clusters_ = len(labels_unique)
    print('聚类的类别数目：%d' % n_clusters_)
    class_num = get_class_num(cluster)
    print('聚类结果为：')
    for label in class_num:
        print(str(label) + ':' + str(class_num[label]))
    write_cluster_result(cluster_result_name, class_num, my_ipc)
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(test_vecs, cluster))
    embedding_file.close()
    label_vecs = get_DBSCAN_clusters(test_vecs, cluster)
    centers = get_centers(label_vecs)
    return model, centers

def keyword_extraction(log_file_name, test_name, wordvec_name, birch_model, centers, dim=100, topn=20):
    log_file = open(log_file_name, 'w', encoding='utf-8')
    wordvec_file = open(wordvec_name, 'r', encoding='utf-8', errors='surrogateescape')
    stopwords = get_stopwords()
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
                    if word not in stopwords and word in word2ind:
                        line_words.append(word)
                        cur_wordvec = wordvecs[word2ind[word]].reshape(1, dim)
                        line_vecs.append(cur_wordvec)
                assert len(line_words) == len(line_vecs)
                ind2vec = get_index2vectors(word2ind, wordvecs, line_words)
                # get_most_label(line_vecs, birch_model)      ####################################
                most_label = get_most_label(line_vecs, birch_model)
                # print(most_label)
                # center = birch_model.subcluster_centers_[most_label]    ##################################
                center = centers[most_label]
                sorted_index_distance = distance_sort(ind2vec, center, 'cos')
                print('-------keyword-------')
                log_file.write('-------keyword-------\n')
                keyword_num = 0
                for item in list(sorted_index_distance.items()):
                    cur_word = words[item[0]]
                    cur_dis = item[1]
                    log_file.write('%s\t\t%f\n' % (cur_word, cur_dis))
                    print(cur_word + '\t' + str(cur_dis))
                    keyword_num += 1
                    if keyword_num >= topn:
                        break
                print('-----------------------------------------------------------------')
                log_file.write('------------------------------------------------------------------\n')
                num += 1
    wordvec_file.close()
    log_file.close()


if __name__ == '__main__':
    sent2vec_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\sent2vec\bxd_fc_rm_techField_100.vec'
    embedding_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_abstract_100_mincount1.vec'
    # embedding_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_techField_100.vec'
    wordvec_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_abstract_100_mincount1.vec'
    test_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\kTVq\_kTVq_label_abstract.txt'
    dbscan_train_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\kTVq\_kTVq_label_techField.txt'
    cluster_result_name = '../data/patent_abstract/DBSCAN/kTVq_techField_wordAVG_keywordTest.txt'
    log_file_name = r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\test\kTVq_techField_wordAVG.txt'
    # dbscan1()
    # dbscan_model, centers = dbscan2(sent2vec_name, dbscan_train_name, cluster_result_name)
    dbscan_model, centers = dbscan3(embedding_name, dbscan_train_name, cluster_result_name)
    # keyword_extraction(log_file_name, test_name, wordvec_name, birch_model, centers)
