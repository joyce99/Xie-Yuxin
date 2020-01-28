import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from pylab import mpl
#
# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def plot_with_labels(low_dim_embs, colors, labels, filename):   # 绘制词向量图
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(20, 20))  # in inches
    colors = ['red', 'blue', 'green', 'black']
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

def read(file, threshold=0, dtype='float'):
    print('读取词向量文件中......')
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype)
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        words.append(word)
        matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
    return (words, matrix)

if __name__ == '__main__':

    try:
        # pylint: disable=g-import-not-at-top
        embedding_file = open('../data/model/SE2010.vector', 'r',encoding='utf-8', errors='surrogateescape')
        words, vectors = read(embedding_file, dtype=float)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 1000
        low_dim_embs = tsne.fit_transform(vectors[:plot_only, :])
        labels = [words[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels, '../data/SE2010.png')

    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)