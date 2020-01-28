import pkuseg
# import jieba


def pku1():
    try:
        f1 = open('D:\PycharmProjects\Dataset\keywordEX\patent\_all_rm_abstract_NEW.txt', 'r', encoding='utf-8')
        f2 = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\all_fc_abstract_NEW.txt', 'w', encoding='utf-8')
        mystr = f1.readlines()
        iters = 1
        seg = pkuseg.pkuseg()  # 以默认配置加载模型
        for line_str in mystr:
            seg_list = seg.cut(line_str)
            result = ' '.join(seg_list)
            f2.write(result)
            print('处理完成%d行' % iters)
            iters += 1
        # print(mystr[0])

    finally:
        if f1:
            f1.close()
        if f2:
            f2.close()

def pku2():
    stopfile = open('../data/patent_abstract/stopwords_new.txt', 'r', encoding='utf-8')
    f1 = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\bxd\_bxd_techField.txt', 'r', encoding='utf-8')
    f2 = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\bxd\bxd_fc_rm_techField_PKU.txt', 'w', encoding='utf-8')
    mystr = f1.readlines()
    stopwords = list()
    for line in stopfile.readlines():
        stopwords.append(line.strip())
    iters = 1
    seg = pkuseg.pkuseg()  # 以默认配置加载模型
    for line_str in mystr:
        seg_list = seg.cut(line_str)
        words = [word for word in seg_list if word not in stopwords]
        result = ' '.join(words)
        result += '\n'
        f2.write(result)
        print('处理完成%d行' % iters)
        iters += 1
    stopfile.close()
    f1.close()
    f2.close()

if __name__ == '__main__':
    pku2()
    # seg = pkuseg.pkuseg()  # 以默认配置加载模型
    # seg_list = seg.cut('我爱北京\n')
    # print(seg_list)

