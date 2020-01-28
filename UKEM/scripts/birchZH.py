import re
import jieba
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import operator
from embeddings import read
from sklearn.cluster import Birch
from sklearn import metrics
from textrank4zh import TextRank4Keyword
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from rake import Rake


class patent_ZH:
    def __init__(self, content, doc_num, ipc):
        self.label = -1
        self.content = content
        self.doc_num = doc_num
        self.docvec = None
        self.ipc = ipc

def get_Birch_clusters(vectors,labels, dim=100):    # 根据Birch聚类后的标签labels整理各类的向量，存放在字典clusters
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
        if label == -1:  # 如果是噪音类
            continue
        else:
            cur_vectors = clusters[label]
            cur_center = np.mean(cur_vectors, axis=0).reshape(1, dim)
            # print(cur_vectors.shape)
            # print(np.mean(cur_vectors, axis=0).shape)
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

def get_stopwords(fname):
    stop_file = open(fname, 'r', encoding='utf-8')
    stopwords = list()
    for line in stop_file.readlines():
        stopwords.append(line.strip())
    stop_file.close()
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

def get_PKEA_cluster_center(test_name, wordvecs, word2ind, dim=100):       #返回PKEA算法的某个类的预先定义单词的那一个聚类中心
    pre_train_vecs = np.zeros((1, dim))
    PKEA_cluster_center = {
        'F24F': ['空调', '水泵', '过滤网', '冷凝器', '管路', '氟利昂', '变频', '蒸发器', '通风口', '空气'],
        'H04N': ['数字电视', '视觉', '液晶屏幕', '机顶盒', '显像管', '监测器', '视频', '解码', '影像', '电信号'],
        'B08B': ['清洁', '清洗', '擦拭', '软管', '除尘', '清洗机', '超声波', '毛刷', '喷淋', '洗刷'],
        'F25D': ['冰箱', '蒸发器', '冷凝器', '压缩机', '通风孔', '温度传感器', '气流', '除霜', '隔板', '制冷系统'],
        'D06F': ['洗衣机', '加热器', '烘干', '自动化', '臭氧', '离子', '磨损', '脱水', '排水管', '活性氧'],
        'H04M': ['电话', '终端', '联系人', '安全控制', '续航', '电话号码', '智能手机', '通讯', '光纤', '交换机']
    }
    if re.search('kongtiao', test_name):
        cluster_id = 'F24F'
    elif re.search('TV', test_name):
        cluster_id = 'H04N'
    elif re.search('qingjie', test_name):
        cluster_id = 'B08B'
    elif re.search('bingxiang', test_name):
        cluster_id = 'F25D'
    elif re.search('xiyiji', test_name):
        cluster_id = 'D06F'
    elif re.search('phone', test_name):
        cluster_id = 'H04M'
    for word in PKEA_cluster_center[cluster_id]:
        word_index = word2ind[word]
        cur_vec = wordvecs[word_index].reshape(1, dim)
        pre_train_vecs = np.row_stack((pre_train_vecs, cur_vec))
    pre_train_vecs = np.delete(pre_train_vecs, 0, 0)
    km_model = KMeans(n_clusters=1, init='k-means++', max_iter=1000)
    km_model.fit(pre_train_vecs)
    center = km_model.cluster_centers_[0]
    return center


# def get_most_label(line_vecs, birch_model):       #摘要中哪个类的单词数目最多，就归为哪一类
#     label_num = dict()
#     for vec in line_vecs:
#         cur_label = birch_model.predict(vec)
#         if cur_label[0] not in label_num:
#             label_num[cur_label[0]] = 1
#         else:
#             label_num[cur_label[0]] += 1
#     label_num = dict(sorted(label_num.items(), key=operator.itemgetter(1), reverse=True))
#     most_label = list(label_num.items())[0][0]
#     return most_label
def get_most_label(line_vecs, birch_model, dim=100):    #词向量加和
    line_matrix = np.zeros((1, dim))
    for vec in line_vecs:
        line_matrix = np.row_stack((line_matrix, vec))
    line_matrix = np.delete(line_matrix, 0, 0)
    line_AVG = np.mean(line_matrix, axis=0).reshape(1, dim)
    most_label = birch_model.predict(line_AVG)
    return most_label

def frequency_test(line_words, topn=20):      #frequency算法，返回一个字典{关键词：频率}
    frequency_dict = dict()
    for word in line_words:
        if word not in frequency_dict:
            frequency_dict[word] = 1
        else:
            frequency_dict[word] += 1
    frequency_sorted  = dict(sorted(frequency_dict.items(), key=operator.itemgetter(1), reverse=True))
    return dict(list(frequency_sorted.items())[0 : min(topn, len(frequency_sorted)) : 1])

def mytfidf(test_name, stopwords, keywordstop, topn=20):     #TF-IDF算法，返回字典{index:[关键字1,关键字2...]}
    corpus = list()
    tfidf_keywords = dict()
    with open(test_name, 'r', encoding='utf-8') as test_file:
        num = 0
        for test_line in test_file.readlines():
            line_split = test_line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                print('第%d条专利摘要：' % (num + 1))
                print(content)
                test_line_words = list(jieba.cut(content))
                line_words = list()
                for word in test_line_words:
                    if word not in stopwords and word not in keywordstop and len(word)>1 and not word.isdigit():
                        line_words.append(word)
                line_str = ' '.join(line_words)
                corpus.append(line_str)
                num += 1
    vectorizer = CountVectorizer(lowercase=False)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for line_index in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        weight_index = dict()
        for word_index in range(len(word)):
            weight_index[weight[line_index][word_index]] = word_index
        sorted_weight_index = dict(sorted(weight_index.items(), key=operator.itemgetter(0), reverse=True)[0 : min(topn, len(word)) : 1])
        line_keywords = list()
        for keyword_weight in sorted_weight_index:
            line_keywords.append(word[sorted_weight_index[keyword_weight]])
            # print(word[sorted_weight_index[keyword_weight]], keyword_weight)
        tfidf_keywords[line_index] = line_keywords
    return tfidf_keywords

def birch3(embedding_name, birch_train_name, cluster_result_name):       # 词向量加和平均
    embedding_file = open(embedding_name, 'r', encoding='utf-8', errors='surrogateescape')
    patent_list = list()
    dim = 100
    stopwords = get_stopwords('../data/patent_abstract/stopwords_new.txt')
    words, wordvecs = read(embedding_file, dtype=float)
    word2ind = {word: i for i, word in enumerate(words)}
    test_vecs = np.zeros((1, dim))
    with open(birch_train_name, 'r', encoding='utf-8') as test_file:
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
                # cur_linevec = np.mean(line_wordvecs, axis=0).reshape(1, dim)
                # cur_patent.docvec = cur_linevec
                # patent_list.append(cur_patent)
                # test_vecs = np.row_stack((test_vecs, cur_linevec))
                # print('处理第%d条专利......' % (num+1))
                if line_wordvecs.all() == 0:
                    continue
                else:
                    cur_linevec = np.mean(line_wordvecs, axis=0).reshape(1, dim)
                    cur_patent.docvec = cur_linevec
                    patent_list.append(cur_patent)
                    test_vecs = np.row_stack((test_vecs, cur_linevec))
                    print('处理第%d条专利......' % (num+1))
            num += 1
        test_vecs = np.delete(test_vecs, 0 , 0)
    print(test_vecs.shape)
    model = Birch(threshold=1.45, branching_factor=50, n_clusters=None).fit(test_vecs)
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
    # write_cluster_result(cluster_result_name, class_num, my_ipc)
    # print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(test_vecs, cluster))
    embedding_file.close()
    label_vecs = get_Birch_clusters(test_vecs, cluster)
    centers = get_centers(label_vecs)
    return model, centers

def keyword_extraction(log_file_name, test_name, wordvec_name, birch_model, centers, dim=100, topn=20):
    log_file = open(log_file_name, 'w', encoding='utf-8')
    wordvec_file = open(wordvec_name, 'r', encoding='utf-8', errors='surrogateescape')
    stopwords = get_stopwords('../data/patent_abstract/stopwords_new.txt')
    keywordstop = get_stopwords('../data/patent_abstract/mystop.txt')
    words, wordvecs = read(wordvec_file, dtype=float)
    word2ind = {word: i for i, word in enumerate(words)}
    tfidf_keywords = mytfidf(test_name, stopwords, keywordstop, topn=topn)
    with open(test_name, 'r', encoding='utf-8') as test_file:
        num = 0
        for test_line in test_file.readlines():
            line_split = test_line.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                print('第%d条专利摘要：' % (num+1))
                print(content)
                log_file.write('第%d条专利摘要：\n' % (num + 1))
                log_file.write('%s\n' % content)
                rake = Rake()
                rake_keywords = rake.run(content)
                test_line_words = list(jieba.cut(content))
                line_words = list()
                line_vecs = list()
                for word in test_line_words:
                    if word not in stopwords and word not in keywordstop and word in word2ind and len(word)>1 and not word.isdigit():
                        line_words.append(word)
                        cur_wordvec = wordvecs[word2ind[word]].reshape(1, dim)
                        line_vecs.append(cur_wordvec)
                assert len(line_words) == len(line_vecs)
                # frequency_result = frequency_test(line_words, topn=topn)
                ind2vec = get_index2vectors(word2ind, wordvecs, line_words)
                most_label = get_most_label(line_vecs, birch_model)
                center = centers[most_label]
                # center = centers[1]
                PKEA_center = get_PKEA_cluster_center(test_name, wordvecs, word2ind)
                sorted_index_distance = distance_sort(ind2vec, center, 'cos')
                PKEA_sorted_index_distance = distance_sort(ind2vec, PKEA_center, 'cos')
                keyword_num = 0
                tr4w = TextRank4Keyword(stop_words_file = '../data/patent_abstract/mystop.txt')
                tr4w.analyze(text=content, lower=False, window=3, vertex_source = 'words_no_stop_words', pagerank_config={'alpha': 0.85})
                print('RAKE----TF-IDF----textrank----PKEA----ours-----------------')
                log_file.write('RAKE----TF-IDF----textrank----PKEA----ours-----------------\n')
                for rake_word, tfidf_keyword, textrank_item, PKEA_item, our_item in zip(rake_keywords, tfidf_keywords[num], tr4w.get_keywords(20, word_min_len=2), list(PKEA_sorted_index_distance.items()), list(sorted_index_distance.items())):
                    textrank_word = textrank_item.word
                    PKEA_word = words[PKEA_item[0]]
                    PKEA_dis = PKEA_item[1]
                    our_word = words[our_item[0]]
                    our_dis = our_item[1]
                    log_file.write('%s\t\t%s\t\t%s\t\t%s\t\t%s\n' % (rake_word, tfidf_keyword, textrank_word, PKEA_word, our_word))
                    print(rake_word + '\t\t'+tfidf_keyword + '\t\t'+'\t\t' + textrank_word + '%f' % textrank_item.weight + PKEA_word + '%f' % PKEA_dis + '\t\t'+ '\t\t' + our_word + '%f' % our_dis)
                    keyword_num += 1
                    if keyword_num >= topn:
                        break
                print('------------------------------------------------------------------')
                log_file.write('------------------------------------------------------------------\n')
                num += 1
    wordvec_file.close()
    log_file.close()


if __name__ == '__main__':
    embedding_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_abstract_100_mincount1.vec'
    birch_train_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\bxk\_bxk_label_title.txt'
    cluster_result_name = '../data/patent_abstract/Birch/bxk_title_wordAVG_keywordTest_1.45_50.txt'
    log_file_name = r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\6种专利摘要各100条已标注bxk\bingxiang_RAKE_TFIDF_textRank_PKEA_ours_TITLE_wordAVG_1.45_50.txt'
    test_name = '../data/patent_abstract/6种专利摘要各100未标注/_bingxiang_abstract.txt'
    wordvec_name = r'D:\PycharmProjects\Dataset\keywordEX\patent\word2vec\all_rm_abstract_100_mincount1.vec'
    birch_model, centers = birch3(embedding_name, birch_train_name, cluster_result_name)
    keyword_extraction(log_file_name, test_name, wordvec_name, birch_model, centers)
# def birch1(model_name):       # Doc2vec
#     dim = 100
#     model = Doc2Vec.load(model_name)
#     patent_list = list()
#     docvecs = np.zeros((1, dim))
#     num = 0
#     stopwords = get_stopwords('../data/patent_abstract/stopwords_new.txt')
#     with open('D:\PycharmProjects\Dataset\keywordEX\patent\_bxk_label_abstract.txt', 'r', encoding='utf-8') as curf:
#         for line in curf.readlines():
#             line_split = line.split(' ::  ')
#             if len(line_split) == 2:
#                 content_rm = line_split[1].strip()
#                 line_cut = list(jieba.cut(content_rm))
#                 line_words = [word for word in line_cut if word not in stopwords]
#                 content = line_split[1].strip()
#                 cur_patent = patent_ZH(content, num, line_split[0])
#                 cur_docvec = model.infer_vector(line_words)
#                 cur_patent.docvec = cur_docvec
#                 print('读取第%d个专利摘要......' % (num + 1))
#                 if num == 0:
#                     docvecs[0] = cur_docvec.reshape(1, dim)
#                 else:
#                     docvecs = np.row_stack((docvecs, cur_docvec.reshape(1, dim)))
#                 patent_list.append(cur_patent)
#                 num += 1
#     print(docvecs.shape)
#     model = Birch(n_clusters=3, threshold=0.5, branching_factor=50).fit(docvecs)
#     cluster = model.labels_
#     patent_list = get_label(patent_list, cluster)
#     my_ipc = get_patent_ipc(patent_list)
#     labels_unique = np.unique(cluster)
#     n_clusters_ = len(labels_unique)
#     print('聚类的类别数目：%d' % n_clusters_)
#     class_num = get_class_num(cluster)
#     print('聚类结果为：')
#     for label in class_num:
#         print(str(label) + ':' + str(class_num[label]))
#     write_cluster_result('../data/patent_abstract/Birch/bxk_abstract_nostop_doc2vecTest_100.txt', class_num, my_ipc)
#     print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(docvecs, cluster))
#     return model
#
# def birch2(embedding_name, birch_train_name, cluster_result_name):       # sent2vec
#     embedding_file = open(embedding_name, 'r', encoding='utf-8', errors='surrogateescape')
#     sent_num, sentvecs = read(embedding_file, dtype=float)
#     patent_list = list()
#     num = 0
#     dim = 100
#     with open(birch_train_name, 'r', encoding='utf-8') as curf:
#         for line in curf.readlines():
#             line_split = line.split(' ::  ')
#             if len(line_split) == 2:
#                 content = line_split[1].strip()
#                 cur_patent = patent_ZH(content, num, line_split[0])
#                 cur_patent.docvec = sentvecs[num].reshape(1, dim)
#                 patent_list.append(cur_patent)
#                 print('读取第%d个专利摘要......' % (num + 1))
#                 num += 1
#     print(sentvecs.shape)
#     model = Birch(threshold=0.5, branching_factor=50).fit(sentvecs)
#     cluster = model.labels_
#     patent_list = get_label(patent_list, cluster)
#     my_ipc = get_patent_ipc(patent_list)
#     labels_unique = np.unique(cluster)
#     n_clusters_ = len(labels_unique)
#     print('聚类的类别数目：%d' % n_clusters_)
#     class_num = get_class_num(cluster)
#     print('聚类结果为：')
#     for label in class_num:
#         print(str(label) + ':' + str(class_num[label]))
#     write_cluster_result(cluster_result_name, class_num, my_ipc)
#     print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(sentvecs, cluster))
#     embedding_file.close()
#     label_vecs = get_Birch_clusters(sentvecs, cluster)
#     centers = get_centers(label_vecs)
#     return model, centers