import numpy as np
import re
import os
from gensim.models.doc2vec import Doc2Vec
import jieba
import operator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn import metrics
from embeddings import read, plot_with_labels
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
import scipy
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from sklearn.datasets.samples_generator import make_blobs
from extractTrain import myfile
import csv
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from rake import Rake

rake = Rake()
text = '本发明公开一种具有语音交互功能的声控空调器，通过用户发出的语音指令信息直接对空调器进行控制，并在对空调进行语音控制过程中通过反馈语音指令信息给用户确认，实现用户与空调的语音交互。该技术方案能够完全摆脱遥控器实现对空调的控制，操作方便，同时，语音交互方式具有灵活性，能够满足不同用户个性化的要求，提高了用户的体验。'
keywords = rake.run(text)
print(keywords)

# corpus=["我 来到 北京 清华大学",#第一类文本切词后的结果，词之间以空格隔开
# 		"他 来到 了 网易 杭研 大厦",#第二类文本的切词结果
# 		"小明 硕士 毕业 与 中国 科学院",#第三类文本的切词结果
# 		"我 爱 北京 天安门"]#第四类文本的切词结果
# vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
# transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
# tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
# word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
# weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
# for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
#     print ("-------这里输出第",i,u"类文本的词语tf-idf权重------")
#     for j in range(len(word)):
#         print (word[j],weight[i][j])

#
#
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码
# if __name__ == '__main__':
#     db = pymysql.connect("localhost", "root", "", "patent_system")
#     # 使用 cursor() 方法创建一个游标对象 cursor
#     cursor = db.cursor()
#     sql1 = """SELECT * FROM tb_patentall_label where (label LIKE '%H04M%') OR (label LIKE '%F25D%') OR (label LIKE '%D06F%')"""
#     try:
#         # 执行sql语句
#         cursor.execute(sql1)
#         # 提交到数据库执行
#         # 获取所有记录列表
#         results = cursor.fetchall()
#         patent_id = 0
#         for row in results:
#             my_dict = dict()
#             patent_id += 1
#             my_dict['id'] = patent_id
#             my_dict['app_num'] = pymysql.escape_string(row[1])
#             my_dict['title'] = pymysql.escape_string(row[2])
#             my_dict['abstract'] = pymysql.escape_string(row[3])
#             my_dict['company_name'] = pymysql.escape_string(row[4])
#             my_dict['content'] = pymysql.escape_string(row[5])
#             my_dict['tech_field'] = pymysql.escape_string(row[6])
#             my_dict['tech_bg'] = pymysql.escape_string(row[7])
#             my_dict['label'] = pymysql.escape_string(row[8])
#
#             # SQL 插入语句
#             sql2 = """INSERT INTO tb_patent_bxd_label(id, app_num, title, abstract, company_name, content, tech_field, tech_bg, label)
#                      VALUES ({id}, '{app_num}', '{title}', '{abstract}', '{company_name}', '{content}', '{tech_field}', '{tech_bg}', '{label}')""".format(**my_dict)
#             cursor.execute(sql2)
#             print('插入第%d条专利......' % patent_id)
#         db.commit()
#     except IndexError as e:
#         # 如果发生错误则回滚
#         db.rollback()
#         print(e)
#
#
#     # 关闭数据库连接
#     db.close()
    # log_file.close()
# a = np.load(r'D:\PycharmProjects\Dataset\vocabEN-ES.npy')
# print(a)
# print(a.shape)
# train_f = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\ydy\_0kt1tv_abstract.csv', 'r', encoding='utf-8-sig')
# myreader = csv.DictReader(train_f)
# rows = [row for row in myreader]
# print(rows[0]['label'])
# train_f.close()
# class file_EN:
#     def __init__(self, name):
#         self.name = ""
#         self.label = -1
#         self.doc_num = None
#         self.docvec = None
#
# def search(folder, filters, allfile):
#     folders = os.listdir(folder)
#     for name in folders:
#         curname = os.path.join(folder, name)
#         isfile = os.path.isfile(curname)
#         if isfile:
#             for filter in filters:
#                 if name.startswith(filter):
#                     cur = myfile()
#                     cur.name = name
#                     allfile.append(cur.name)
#                     break
#         else:
#             search(curname, filters, allfile)
#     return allfile
#
# folder = r"../data/SemEval2010/mine"
# filters = ['C','H','I','J']
# allfile = []
# allfile = search(folder, filters, allfile)
# file_len = len(allfile)
# print('共查找到%d个摘要文件' %(file_len))
# train_file = open('../data/SemEval2010/new_line_doc.txt', 'w', encoding='utf-8')
# i = 0
# truth = {'I':[], 'J':[], 'H':[], 'C':[]}
# for f in allfile:
#     for name_start in truth:
#         if f.startswith(name_start):
#             with open(os.path.join(folder, f), 'r', encoding='utf-8') as curf:
#                 for line in curf.readlines():
#                     train_file.write(re.sub('\n', ' ', line))
#             train_file.write('\n')
#             truth[name_start].append(i)
#             i += 1
#             break
# train_file.close()
# print(truth)
# for label in truth:
#     print(label + ':' + str(len(truth[label])))
# print(allfile.sort())
# truth = {'C':[], 'H':[], 'I':[], 'J':[]}
# num = 0
# file_list = []
# # train_file = open('../data/SemEval2010/new_line_doc.txt', 'w', encoding='utf-8')
# for name_start in filters:
#     for i in range(100):
#         cur_name = name_start + '-' + str(i) + '.txt.final'
#         abs_name = os.path.join(folder, cur_name)
#         isfile = os.path.isfile(abs_name)
#         if isfile:
#             # with open(abs_name, 'r', encoding='utf-8') as curf:
#                 # for line in curf.readlines():
#                     # train_file.write(re.sub('\n', ' ', line))
#             # train_file.write('\n')
#             cur_file = file_EN()
#             truth[name_start].append(num)
#             num += 1
# # train_file.close()
# print(truth)

# X1, y1 = datasets.make_blobs(n_samples=100, n_features=2, centers=[[0.5,0.5]], cluster_std=[[.1]],
#                random_state=9)
# X2, y2 = datasets.make_blobs(n_samples=100, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#                random_state=9)
# X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
#                                           noise=.05)
# X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#                random_state=9)
# points = np.concatenate((X1, X2))
# points=scipy.randn(10000,50)
#1. 层次聚类
#生成点与点之间的距离矩阵,这里用的欧氏距离:
# disMat = sch.distance.pdist(points,'euclidean')
#进行层次聚类:
# Z=sch.linkage(points,method='average', metric='euclidean')
# #将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
#
# #根据linkage matrix Z得到聚类结果:
# cluster= sch.fcluster(Z, 0.4, 'distance', depth=2)
# # P=sch.dendrogram(Z)
# # plt.savefig('plot_dendrogram.png')
# model=AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='average')
# cluster = model.fit_predict(points)
# n_clusters_ = len(set(cluster))
# print('聚类的类别/数目：%d' % (n_clusters_))
# print("Original cluster by hierarchy clustering:\n",cluster)
# plt.scatter(points[:, 0], points[:, 1], c=cluster)
# plt.show()

# #2. MeanShift
# ##带宽，也就是以某个点为核心时的搜索半径
# bandwidth = estimate_bandwidth(points, quantile=0.2, n_samples=500)
# ##设置均值偏移函数
# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# ##训练数据
# ms.fit(points)
# ##每个点的标签
# labels = ms.labels_
# print(labels)
# ##簇中心的点的集合
# cluster_centers = ms.cluster_centers_
# ##总共的标签分类
# labels_unique = np.unique(labels)
# ##聚簇的个数，即分类的个数
# n_clusters_ = len(labels_unique)
# plt.scatter(points[:, 0], points[:, 1], c=labels)
# plt.show()

# #3. 层次聚类
# ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
# cluster = ac.fit_predict(points)
# labels_unique = np.unique(cluster)
# n_clusters_ = len(labels_unique)
# print('聚类的类别数目：%d' % n_clusters_)
# X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
#                                           noise=.05)
# X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
#                random_state=9)
# X = np.concatenate((X1, X2))
# y_pred = [-1 for i in range(6000)]
# # plt.scatter(X[:, 0], X[:, 1], marker='o',c=y_pred)
# # plt.show()
# # y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
# y_pred = DBSCAN(eps=0.1, min_samples=10).fit_predict(X)
# print(y_pred.shape)
# n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)
# print('聚类的类别数目：%d' % (n_clusters_))
# ratio = len(y_pred[y_pred[:] == -1]) / len(y_pred)
# print('认为是噪音的数据比例：%d' % (ratio))
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# plt.show()

# 4.BIRCH
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]
# X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3],
#                   random_state =9)
# from sklearn.cluster import Birch
# model = Birch(n_clusters = None, threshold = 0.5, branching_factor = 60).fit(X)
# y_pred = model.labels_
# centers = model.subcluster_centers_
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# for i in range(len(centers)):
#     plt.scatter(centers[i, 0], centers[i, 1], s=80, c='k')
# plt.show()
# from sklearn import metrics
# print ("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred))

# # 验证
# model = Doc2Vec.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10_2.model')
# num = 0
# docvecs = np.zeros((1, 100))
# with open('../data/patent_abstract/_bxk_abstract.txt', 'r', encoding='utf-8') as curf:
#     for line in curf.readlines():
#         content = re.sub('[，。；、]+', '', line)
#         content = content.strip()
#         each_cut = list(jieba.cut(content))
#         line = line.strip()
#         cur_docvec = model.infer_vector(each_cut)
#         print('读取第%d个专利摘要......' % (num + 1))
#         if num == 0:
#             docvecs[0] = cur_docvec
#         else:
#             docvecs = np.row_stack((docvecs, cur_docvec.reshape(1, 100)))
#         num += 1
# word_list = list(jieba.cut('该技术方案能够完全摆脱遥控器实现对空调的控制，操作方便，同时，语音交互方式具有灵活性，能够满足不同用户个性化的要求，提高了用户的体验'))
# print(word_list)
# vector1 = model.infer_vector(word_list)
# vector2 = np.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10.model.docvecs.vectors_docs.npy')
# print(vector1.shape)
# print(vector2.shape)
# sims = model.docvecs.most_similar([vector1], topn=10)
# sims = model.docvecs.most_similar([docvecs[100]], topn=10)
# for i, sim in sims:
#     print(i, sim)
# for sim in sims:
#     print(sim[0])
# print(vector1)

# model = Doc2Vec.load(r'D:\PycharmProjects\KeywordExtraction\data\model\sen2vec\SE2010\SEdoc_50_dm_40.model')
# vec = np.load(r'D:\PycharmProjects\KeywordExtraction\data\model\sen2vec\SE2010\SEdoc_50_dm_40.vector.npy')

# vector1 = model.infer_vector(word_list)
# vector2 = np.load(r'D:\PycharmProjects\Dataset\keywordEX\patent\doc2vec\all_100_dm_10.model.docvecs.vectors_docs.npy')
# print(vector1.shape)
# print(vector2.shape)
# # sims = model.docvecs.most_similar([vector1], topn=10)
# sims = model.docvecs.most_similar([vector2[0]], topn=10)
# for i, sim in sims:
#     print(i, sim)
# # for sim in sims:
# #     print(sim[0])
# print(vector1)

# print(vector2[0])
# num = float(np.dot(vector2[0], vector1.reshape(1, 100).T))
# vec_norm = np.linalg.norm(vector1) * np.linalg.norm(vector2)
# cos = num / vec_norm
# sim = 0.5 + 0.5 * cos   # 归一化
# print(sim)