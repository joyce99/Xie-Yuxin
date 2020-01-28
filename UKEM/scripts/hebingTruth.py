import re

def get_stopwords(fname):
    stop_file = open(fname, 'r', encoding='utf-8')
    stopwords = list()
    for line in stop_file.readlines():
        stopwords.append(line.strip())
    stop_file.close()
    return stopwords

def get_test_result(test_name, test_num=100):         #获得各类算法的关键字结果，返回一个字典
    our_dict = dict()
    num = 0
    ifwrite = False
    test_file = open(test_name, 'r', encoding='utf-8')
    test_lines = test_file.readlines()
    for test_line in test_lines:
        if test_line == 'RAKE----TF-IDF----textrank----ours-----------------\n':
            ifwrite = True
            num += 1
            if num > test_num:
                break
            our_keywords = list()
        elif test_line == '------------------------------------------------------------------\n':
            ifwrite = False
            our_dict[num] = our_keywords
        else:
            if ifwrite:
                line_split = test_line.split('\t\t\t')
                our_keywords.append(line_split[3].strip())
    test_file.close()
    return our_dict

def get_truth_result(truth_name, get_num=100):       #获得人工标注的关键字结果，返回一个字典{1:[key1,key2...]}
    truth_file = open(truth_name, 'r', encoding = 'utf-8')
    truth_lines = truth_file.readlines()
    truth_dict = dict()
    num = 0
    for truth_line in truth_lines:
        if re.search('keywords:', truth_line):
            num += 1
            if num > get_num:
                break
            keywords = list()
            line_words = truth_line.split('keywords:')[1]
            for word in line_words.split('、'):
                if word.strip() != '':
                    keywords.append(word.strip())
            truth_dict[num] = keywords
            print('第%d条人工标注专利关键字提取完成......' % num)
        elif re.search('keywords: ',truth_line):
            num += 1
            if num > get_num:
                break
            keywords = list()
            line_words = truth_line.split('keywords: ')[1]
            for word in line_words.split('、'):
                if word.strip() != '':
                    keywords.append(word.strip())
            truth_dict[num] = keywords
            print('第%d条人工标注专利关键字提取完成......' % num)
    truth_file.close()
    return truth_dict

def jiao_truth(truth_dict1, truth_dict2, our_dict):       #获得交叉验证后的结果，返回一个字典{1:[key1,key2...]}
    keywordstop = get_stopwords('../data/patent_abstract/mystop.txt')
    truth_zonghe = dict()
    assert len(truth_dict1) == len(truth_dict2)
    for num in truth_dict1:
        zonghe_words = list()
        for word in truth_dict1[num]:
            if word in truth_dict2[num] and word not in keywordstop and len(word)>1:
                zonghe_words.append(word)
        zonghe_words = list({}.fromkeys(zonghe_words).keys())
        if len(zonghe_words) < min(10, len(truth_dict1[num]), len(truth_dict2[num])):
            for word in truth_dict1[num]:
                if word in our_dict[num]:
                    zonghe_words.append(word)
                    zonghe_words = list({}.fromkeys(zonghe_words).keys())
                    if len(zonghe_words) >= min(10, len(truth_dict1[num]), len(truth_dict2[num])):
                        break
        truth_zonghe[num] = zonghe_words
    return truth_zonghe

def main():
    truth_name1 = r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\6种专利摘要各100条已标注\移动通信余道远.txt'
    truth_name2 = r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\6种专利摘要各100条已标注\移动通信丁晗.txt'
    our_name = r'..\data\patent_abstract\6种专利摘要各100条已标注\dianhua_RAKE_TFIDF_textRank_ours_techField_wordAVG_1.04_50.txt'
    file_zonghe = open(r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\6种专利摘要各100条已标注\移动通信综合.txt', 'w', encoding='utf-8')
    our_dict = get_test_result(our_name)
    truth_file = open(truth_name1, 'r', encoding='utf-8')
    truth_dict1 = get_truth_result(truth_name1)
    truth_dict2 = get_truth_result(truth_name2)
    truth_zonghe = jiao_truth(truth_dict1, truth_dict2, our_dict)
    truth_lines = truth_file.readlines()
    patent_num = 1
    for truth_line in truth_lines:
        if re.search('keywords:', truth_line):
            file_zonghe.write('keywords:' + '、'.join(truth_zonghe[patent_num]) + '\n')
            patent_num += 1
        else:
            file_zonghe.write(truth_line)
    truth_file.close()
    file_zonghe.close()

if __name__ == '__main__':
    main()