import re
import operator
import numpy as np
from sklearn.cluster import Birch
import jieba
import csv


class patent_ZH:
    def __init__(self, content, doc_num, ipc):
        self.label = -1
        self.content = content
        self.doc_num = doc_num
        self.docvec = None
        self.ipc = ipc

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

def get_stopwords(fname):
    stop_file = open(fname, 'r', encoding='utf-8')
    stopwords = list()
    for line in stop_file.readlines():
        stopwords.append(line.strip())
    return stopwords

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

def get_test_truth(my_ipc):
    truth = dict()
    for cluster_label in my_ipc:
        cur_result_count = dict()
        cur_label_ipcs = my_ipc[cluster_label]
        for cur_ipc in cur_label_ipcs:
            if cur_ipc not in cur_result_count:
                cur_result_count[cur_ipc] = 1
            else:
                cur_result_count[cur_ipc] += 1
        cur_result_count = dict(sorted(cur_result_count.items(), key=operator.itemgetter(1), reverse=True))
        predict_ipc = list(cur_result_count.keys())[0]
        truth[cluster_label] = predict_ipc
    return truth

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

def birch3(embedding_name, birch_train_name, cluster_result_name):       # 词向量加和平均
    embedding_file = open(embedding_name, 'r', encoding='utf-8', errors='surrogateescape')
    patent_list = list()
    dim = 100
    stopwords = get_stopwords('../data/patent_abstract/stopwords_new.txt')
    words, wordvecs = read(embedding_file, dtype=float)
    word2ind = {word: i for i, word in enumerate(words)}
    test_vecs = np.zeros((1, dim))
    with open(birch_train_name, 'r', encoding='utf-8') as csvFile:
        csv_reader = csv.reader(csvFile)
        num = 0
        birth_header = next(csv_reader)
        for row in csv_reader:
            content = row[1].strip()
            cur_patent = patent_ZH(content, num, row[0])
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
            print('处理第%d条专利......' % (num + 1))
            num += 1
    test_vecs = np.delete(test_vecs, 0, 0)
    print('用于聚类的矩阵维度：')
    print(test_vecs.shape)
    model = Birch(threshold=1.009, branching_factor=50, n_clusters=None).fit(test_vecs)
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
    embedding_file.close()

if __name__ == '__main__':
    embedding_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_abstract_100_mincount1.vec'  #词向量文件
    birch_train_name = 'D:\PycharmProjects\Dataset\keywordEX\patent\ydy\_0bx1dh_abstract.csv'   #有标签数据
    cluster_result_name = 're.txt' #写入聚类结果的文件
    birch3(embedding_name, birch_train_name, cluster_result_name)
    my_ipc = dict()
    ipc_num = 0
    with open(cluster_result_name, 'r', encoding='utf-8') as result_f:
        result_lines = result_f.readlines()
        line_num = 0
        if_write = False
        cur_label = -1
        while line_num < len(result_lines):
            search_title = re.search('类标签为:', result_lines[line_num])
            if search_title:
                cur_label = int(result_lines[line_num].split(':')[1])
                if_write = True
                line_num += 2
            if if_write:
                cur_label_ipc = result_lines[line_num].split(':  ')[1].split('   ')[0].strip()
                if cur_label not in my_ipc:
                    my_ipc[cur_label] = [cur_label_ipc]
                    ipc_num += 1
                else:
                    my_ipc[cur_label].append(cur_label_ipc)
                    ipc_num += 1
                line_num += 1
            else:
                line_num += 1
    truth = get_test_truth(my_ipc)
    print('预测的类标签为：')
    print(truth)
    error = 0.0
    for label in truth:
        for label_ipc in my_ipc[label]:
            if label_ipc != truth[label]:
                error += 1
    print('聚类准确率为：%f%%' % (100-error/ipc_num*100))
