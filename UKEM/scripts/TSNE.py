import numpy as np
import operator
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import jieba
from embeddings import read
from sklearn.cluster import Birch
from pylab import mpl


def get_Birch_clusters(vectors,labels, dim=100):    # 根据DBSCAN聚类后的标签labels整理各类的向量，存放在字典clusters
    clusters = dict()
    for i in range(len(labels)):
        if labels[i] not in clusters:
            clusters[labels[i]] = vectors[i].reshape(1, dim)
        elif labels[i] in clusters:
            cur_vec = vectors[i].reshape(1, dim)
            cur_cluster = clusters[labels[i]]
            clusters[labels[i]] = np.row_stack((cur_cluster, cur_vec))
    clusters = dict(sorted(clusters.items(), key=operator.itemgetter(0)))
    return clusters

def get_centers(clusters, dim=100):  # 获得各个类的中心点(噪音类除外)
    centers = np.zeros((len(clusters), dim))
    for label in clusters:
        if label == -1:     #如果是噪音类
            continue
        else:
            cur_vectors = clusters[label]
            cur_center = np.mean(cur_vectors, axis=0).reshape(1, dim)
            centers[label] = cur_center
    return centers

def plot_with_labels(low_dim_embs, color_labels, ipc_labels, filename):
    assert low_dim_embs.shape[0] >= len(color_labels), 'More labels than embeddings'
    # assert len(color_labels) == len(ipc_labels)
    plt.figure(figsize=(19, 19))  # in inches
    color_list = ['b', 'y', 'g']
    for i, label in enumerate(color_labels):
        x, y = low_dim_embs[i, :]
        # if i in range(2687, 2690):
        if i in range(2225, 2228):
            plt.scatter(x, y, s=250, c='r')
        else:
            plt.scatter(x, y, c=color_list[label])
            # plt.annotate(ipc_labels[i], xy=(x, y), xytext=(5, 2), textcoords='offset points',ha='right', va='bottom')
    plt.savefig(filename)

def techField_wordAVG_display():
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    embedding_file = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_abstract_100_mincount1.vec', 'r',encoding='utf-8', errors='surrogateescape')
    stop_file = open('../data/patent_abstract/stopwords_new.txt', 'r', encoding='utf-8')
    stopwords = list()
    dim = 100
    for line in stop_file.readlines():
        stopwords.append(line.strip())
    words, wordvecs = read(embedding_file, dtype=float)
    word2ind = {word: i for i, word in enumerate(words)}
    tsne_vecs = np.zeros((1, dim))
    ipc_list = list()
    with open(r'D:\PycharmProjects\Dataset\keywordEX\patent\bxd\_bxd_label_abstract.txt', 'r', encoding='utf-8') as test_file:
        num = 0
        for test_line in test_file.readlines():
            line_split = test_line.split(' ::  ')
            if len(line_split) == 2:
                ipc_list.append(line_split[0][:4])
                content = line_split[1].strip()
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

                # cur_linevec = np.mean(line_wordvecs, axis=0).reshape(1, dim)
                # tsne_vecs = np.row_stack((tsne_vecs, cur_linevec))
                # print('处理第%d条专利......' % (num + 1))
                if line_wordvecs.all() == 0:
                    continue
                else:
                    cur_linevec = np.mean(line_wordvecs, axis=0).reshape(1, dim)
                    tsne_vecs = np.row_stack((tsne_vecs, cur_linevec))
                    print('处理第%d条专利......' % (num+1))
            num += 1
        tsne_vecs = np.delete(tsne_vecs, 0, 0)
    print(tsne_vecs.shape)
    birch_model = Birch(threshold=0.8, branching_factor=50, n_clusters=None).fit(tsne_vecs)
    cluster = list(birch_model.labels_)
    label_vecs = get_Birch_clusters(tsne_vecs, cluster)
    centers = get_centers(label_vecs)
    tsne_vecs = np.row_stack((tsne_vecs, centers))
    print(tsne_vecs.shape)
    low_dim_embs = tsne.fit_transform(tsne_vecs)
    for i in range(3):
        cluster.append(-2)
    print(len(cluster))
    plot_with_labels(low_dim_embs, cluster, ipc_list, '../data/bxd_abstract_TSNE_cluster.png')
    # plot_with_labels(low_dim_embs, cluster, '../data/bxd_TSNE_cluster_NEW123.png')

if __name__ == '__main__':
    techField_wordAVG_display()


