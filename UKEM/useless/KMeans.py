import re
import jieba
import numpy as np
from gensim.models.doc2vec import Doc2Vec
import operator
from sklearn.cluster import KMeans
from embeddings import read
from sklearn import metrics

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

def kmeans3():       # 词向量加和平均
    embedding_file = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_techField_NEW.vec', 'r',
                          encoding='utf-8', errors='surrogateescape')
    stop_file = open('../data/patent_abstract/stopwords_new.txt', 'r', encoding='utf-8')
    stopwords = list()
    patent_list = list()
    dim = 100
    for line in stop_file.readlines():
        stopwords.append(line.strip())
    words, wordvecs = read(embedding_file, dtype=float)
    word2ind = {word: i for i, word in enumerate(words)}
    test_vecs = np.zeros((1, dim))
    with open('D:\PycharmProjects\Dataset\keywordEX\patent\_bxd_label_techField.txt', 'r', encoding='utf-8') as test_file:
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
    cluster = KMeans(n_clusters=3, init='k-means++', max_iter=10000).fit_predict(test_vecs)
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
    with open('../data/patent_abstract/Kmeans/bxd_techField_word2vecAVG_Test.txt', 'w', encoding='utf-8') as result_f:
        result_f.write('聚类结果为：\n')
        for label in class_num:
            result_f.write(str(label) + ':' + str(class_num[label]) + '\n')
        for label in my_ipc:
            result_f.write('类标签为:' + str(label) + ':' + '\n')
            result_f.write(str(class_num[label]) + '条专利' + '\n')
            for ipc in my_ipc[label]:
                result_f.write(str(label) + ':  ' + ipc + '\n')
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(test_vecs, cluster))
    embedding_file.close()
    stop_file.close()

if __name__ == '__main__':
    # kmeans2()
    kmeans1()
    # kmeans3()
    # with open('../data/patent_abstract/Kmeans/Test.txt', 'r', encoding='utf-8') as result_f:
    #     result_lines = result_f.readlines()
    #     line_num = 0
    #     if_write = False
    #     while line_num < len(result_lines):
    #         search_title = re.search('类标签为:', result_lines[line_num])
    #         if search_title:
    #             cur_label = result_lines[line_num].split(':')[1]
    #             if_write = True
    #             line_num += 2
    #         elif if_write:


