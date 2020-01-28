import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# from  result_test import main
# x = [i for i in range(5,21)]
# RAKE_y, TF_IDF_y, textRank_y, PKEA_y, ours_y = main()
# assert len(x) == len(RAKE_y)
# fig = plt.figure()
# plt.plot(x, RAKE_y, marker='D', label='RAKE', lw=2)
# plt.plot(x, TF_IDF_y, marker='^', label='TF-IDF', lw=2)
# plt.plot(x, textRank_y, marker='x', label='TextRank', lw=2)
# plt.plot(x, PKEA_y, marker='|', label='PKEA', lw=2)
# plt.plot(x, ours_y, marker='o', label='ours', lw=2)
# def to_percent(temp, position):
#     return '%1.0f'%(temp) + '%'
# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
# plt.xlabel('The number of keywords extracted by algorithms')
# plt.ylabel('F1 score')
# plt.legend(loc='lower right')
# plt.ylim((25, 60))
# plt.grid(axis="y")
# # filename = r'D:\PycharmProjects\KeywordExtraction\data\bxk_TITLE_F1_line_1.45.png'
# # plt.savefig(filename)
# plt.show()

name_list = ['Group 1', 'Group 2', 'Group 3']
abstract = [48.82, 47.04, 45.84]
title = [56.85, 58.22, 57.40]
techfield = [60.86, 59.78, 56.84]
x1 = list(range(len(abstract)))
x2 = [i + 0.2 for i in x1]
x3 = [i + 0.2 for i in x2]
width = 0.2
# total_width, n = 0.6, 3
# width = total_width / n
def to_percent(temp, position):
     return '%1.0f'%(temp) + '%'
plt.xlabel('Patent clustering data')
plt.ylabel('F1 score')
plt.grid(alpha=0.3,axis="y")
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.bar(x1, abstract, width=width, label='abstract', fc = 'y')
plt.bar(x2, title, width=width, label='title', fc = 'r')
plt.bar(x3, techfield, width=width, label='technical field',fc = 'b')
plt.xticks([i+0.2 for i in x1], name_list)
plt.ylim((35, 70))
plt.legend()
filename = r'D:\PycharmProjects\KeywordExtraction\data\柱状图New.png'
plt.savefig(filename)
plt.show()



# import re
# import random
#
# f_read = open('D:\PycharmProjects\Dataset\spambase.arff', 'r', encoding='utf-8')
# lines = f_read.readlines()
# ifwrite = False
# num_zero = 0
# num_one = 0
# i = 0
# train_dict = {0:[], 1:[]}
# test_dict = {0:[], 1:[]}
# print(len(lines))
# for line in lines:
#     if re.search('@data', line):
#         ifwrite = True
#         continue
#     if ifwrite:
#         i += 1
#         print(line)
#         cur_split = line.split(',')
#         if cur_split[57].strip() == '0':
#             if len(train_dict[0]) < 2230:
#                 train_dict[0].append(line.strip())
#             else:
#                 test_dict[0].append(line.strip())
#             num_zero += 1
#         elif cur_split[57].strip() == '1':
#             if len(train_dict[1]) < 1451:
#                 train_dict[1].append(line.strip())
#             else:
#                 test_dict[1].append(line.strip())
#             num_one += 1
# # print(num_zero)
# # print(num_one)
# # print(i)
# # print(len(train_dict[0]),len(train_dict[1]))
# # print(len(test_dict[0]),len(test_dict[1]))
# f_read.close()
# fw_train = open(r'D:\PycharmProjects\Dataset\train.txt', 'w', encoding='utf-8')
# fw_test = open(r'D:\PycharmProjects\Dataset\test.txt', 'w', encoding='utf-8')
# # train = train_dict[0][:446]
# # train.extend(train_dict[1][:290])
# # random.shuffle(train)
# test = test_dict[0][:111]
# test.extend(test_dict[1][:72])
# random.shuffle(test)
# # print(len(train))
# # print(len(test))
# for line in train:
#     fw_train.write(line + '\n')
# for line in test:
#     fw_test.write(line + '\n')
# train_un_dict = {0:[], 1:[]}
# test_un_dict = {0:[], 1:[]}
# for row in range(446,2230):
#     cur_str = train_dict[0][row].split(',')
#     cur_str[57] = 'unknown'
#     train_un_dict[0].append(','.join(cur_str))
# for row in range(290,1451):
#     cur_str = train_dict[1][row].split(',')
#     cur_str[57] = 'unknown'
#     train_un_dict[1].append(','.join(cur_str))
# for row in range(111,558):
#     cur_str = test_dict[0][row].split(',')
#     cur_str[57] = 'unknown'
#     test_un_dict[0].append(','.join(cur_str))
# for row in range(72,362):
#     cur_str = test_dict[1][row].split(',')
#     cur_str[57] = 'unknown'
#     test_un_dict[1].append(','.join(cur_str))
# train_un = train_un_dict[0]
# train_un.extend(train_un_dict[1])
# random.shuffle(train_un)
# test_un = test_un_dict[0]
# test_un.extend(test_un_dict[1])
# random.shuffle(test_un)
# for line in train_un:
#     fw_train.write(line + '\n')
# for line in test_un:
#     fw_test.write(line + '\n')
# fw_train.close()
# fw_test.close()
# print(len(train))
# print(len(test))
# print(len(train_un))
# print(len(test_un))
# # fr_train = open(r'D:\PycharmProjects\Dataset\train_old.txt', 'r', encoding='utf-8')
# # fr_test = open(r'D:\PycharmProjects\Dataset\test_old.txt', 'r', encoding='utf-8')
# # fw_train = open(r'D:\PycharmProjects\Dataset\train.txt', 'w', encoding='utf-8')
# # fw_test = open(r'D:\PycharmProjects\Dataset\test.txt', 'w', encoding='utf-8')
# #
# # fr_train.close()
# # fr_test.close()
# # fw_train.close()
# # fw_test.close()