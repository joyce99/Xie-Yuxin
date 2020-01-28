import jieba

def jieba1():
    try:
        f1 = open('D:\PycharmProjects\Dataset\keywordEX\patent\_qingjie_abstract.txt', 'r', encoding='utf-8')
        f2 = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\6种专利摘要各100未标注\qingjie.txt', 'w', encoding='utf-8')
        mystr = f1.readlines()
        iters = 1
        for line_str in mystr:
            # seg_list = jieba.cut(line_str)
            # result = ' '.join(seg_list)
            f2.write(line_str)
            f2.write('keywords:\n\n')
            print('处理完成%d行' % iters)
            iters += 1
        # print(mystr[0])

    finally:
        if f1:
            f1.close()
        if f2:
            f2.close()

def jieba2():
    try:
        stopfile = open('../data/patent_abstract/stopwords_new.txt', 'r', encoding='utf-8')
        f1 = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\kTVq\_kTVq_abstract.txt', 'r', encoding='utf-8')
        f2 = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\kTVq\kTVq_fc_rm_abstract.txt', 'w', encoding='utf-8')
        mystr = f1.readlines()
        stopwords = list()
        for line in stopfile.readlines():
            stopwords.append(line.strip())
        iters = 1
        for line_str in mystr:
            seg_list = list(jieba.cut(line_str))
            words = [word for word in seg_list if word not in stopwords]
            result = ' '.join(words)
            f2.write(result)
            print('处理完成%d行' % iters)
            iters += 1

    finally:
        if stopfile:
            stopfile.close()
        if f1:
            f1.close()
        if f2:
            f2.close()

if __name__ == '__main__':
    jieba2()