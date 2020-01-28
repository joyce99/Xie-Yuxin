import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn import metrics
from embeddings import read, plot_with_labels
from sklearn.manifold import TSNE


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

def get_centers(clusters):  # 获得各个类的中心点(噪音类除外)
    centers = {}
    for label in clusters:
        if label == -1:     #如果是噪音类
            continue
        else:
            cur_vectors = clusters[label]
            km_model = KMeans(n_clusters=1, max_iter=500, random_state=0).fit(cur_vectors)
            km_labels = km_model.labels_
            km_score = metrics.calinski_harabaz_score(cur_vectors, km_labels)
            print('类标签为%d的K-means聚类得分：%f' % (label, km_score))
            cur_center = km_model.cluster_centers_
            print('类标签为%d的K-means聚类中心：' %label + str(cur_center))
            centers[label] = cur_center
    return centers

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

def distance_sort(vectors, cur_center, method):     # 获得根据与中心点距离大小排序后的{词向量：与中心点的距离}
    distance_dict = {}
    for vector in vectors:
        distance = get_distance(vector, cur_center, method)
        distance_dict[vector] = distance
    sorted_distance = sorted(distance_dict.items(), key=operator.itemgetter(1))
    sorted_distance_dict = dict(sorted_distance)
    return sorted_distance_dict

def get_vectors(filename, word2ind, wordvecs, dim):    # 获得测试文本中所有词的词向量
    vectors = np.zeros((1, dim))
    test_file = open(filename, 'r', encoding='utf-8')
    for line in test_file.readlines():
        curline_words = line.split(' ')
        for word in curline_words:
            if word == '\n':
                continue
            elif word in word2ind:
                cur_index = word2ind[word]
                cur_vec = wordvecs[cur_index]
                vectors = np.row_stack((vectors,cur_vec))
    if len(vectors) > 1:
        vectors = np.delete(vectors, 0, 0)
    test_file.close()
    return vectors

def get_most_label(test_vectors, clusters):     # 获得测试文本中单词数最多的类别
    class_vector = {}
    for vector in test_vectors:
        for label in clusters:
            if vector in clusters[label]:
                if label not in class_vector:
                    class_vector[label] = vector
                else:
                    class_vector[label] = np.row_stack((class_vector[label], vector))
                break
    assert len(class_vector) > 0
    class_vector = dict(sorted(class_vector.items(), key=operator.itemgetter(0)))
    if len(class_vector) == 1:
        most_label = -1
        print('所有词向量均为噪音！')
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
    print('本文中%d类包含的单词最多，单词数为：%d,占本文单词的%f%%' % (most_label, most_num, most_num * 100.0 / test_vectors.shape[0]))
    return most_label

def main():
    embedding_file = open('../data/model/SE2010.vector', 'r', encoding='utf-8', errors='surrogateescape')
    words, wordvecs = read(embedding_file, dtype=float)
    word2ind = {word: i for i, word in enumerate(words)}
    # vec2ind = {}
    # i = 0
    # for vector in wordvecs:
    #     vec2ind[vector] = i
    #     i += 1
    # vec2ind = {vector: i for vector in wordvecs, for i in range(len(wordvecs))}
    plot_only = 1000
    db_model = DBSCAN(eps=1.79, min_samples=4).fit(wordvecs)
    db_labels = db_model.labels_
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    print('聚类的类别数目(噪音类除外)：%d' % n_clusters)
    ratio = len(db_labels[db_labels[:] == -1]) / len(db_labels)
    print('噪音率:' + str(ratio))
    clusters = get_DBSCAN_clusters(wordvecs, db_labels)
    centers = get_centers(clusters)
    test_vectors = get_vectors('../data/SemEval2010/train_removed/C-41.txt', word2ind, wordvecs, wordvecs.shape[1])
    most_label = get_most_label(test_vectors, clusters)
    vector_distance = distance_sort(test_vectors, centers[most_label], 'ED')
    top_k = 0
    # for vector in vector_distance:
    #     for cur_index in range(len(wordvecs)):
    #         if wordvecs[cur_index] ==
    #         cur_index = vec2ind[vector]
    #         cur_word = wordvecs[cur_index]
    #         top_k += 1
    #         print('%d、%s' % (top_k, cur_word))
    #         if top_k >= 10:
    #             break




            # sorted_distance_dict = distance_sort(clusters[label], centers[label], 'ED') # or cos

    embedding_file.close()




if __name__ == '__main__':
    main()
    # get_vectors('../data/SemEval2010/train_removed/C-41.txt')
    # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    # low_dim_embs = tsne.fit_transform(vectors)
    # # low_dim_embs = tsne.fit_transform(vectors[:plot_only, :])
    # print(low_dim_embs.shape)
    # labels = [words[i] for i in range(vectors.shape[0])]
    # plot_with_labels(low_dim_embs, db_labels, labels, '../data/DBSCAN_SE2010.png')
    # embedding_file.close()

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

