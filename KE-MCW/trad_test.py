import operator
import nltk
import json
from rake import Rake
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from textrank4zh import TextRank4Keyword


def get_kp(content_list, keywords_list):
    kp_list = []
    cur_kp = []
    tmp = keywords_list.copy()
    con_index = 0
    while con_index < len(content_list) and len(tmp) > 0:
        if content_list[con_index] in tmp:
            cur_kp.append(content_list[con_index])
            con_index += 1
            if con_index == len(content_list) and len(cur_kp) > 1:
                str_kp = ' '.join(cur_kp)
                kp_list.append(str_kp)
            continue
        elif 1 < len(cur_kp) < 6:
            str_kp = ' '.join(cur_kp)
            kp_list.append(str_kp)
            for word in cur_kp:
                if word in tmp:
                    delt = tmp.pop(tmp.index(word))
        cur_kp = []
        con_index += 1
    kp_list.extend(keywords_list)
    return kp_list

def get_rake_kp(file_name, topk):
    json_file = open(file_name, 'r', encoding='utf-8')
    rake_kp = []
    for line in json_file.readlines():
        json_data = json.loads(line)
        cur_content = json_data['title'].strip().lower() + ' ' + json_data['abstract'].strip().lower()
        content_list = nltk.word_tokenize(cur_content)
        rake = Rake()
        keywords_dict = rake.run(cur_content)
        keywords_list = list(keywords_dict.keys())[:topk]
        kp_list = get_kp(content_list, keywords_list)
        rake_kp.append(kp_list)
    json_file.close()
    return rake_kp

def get_tfidf_kp(file_name, topk):
    json_file = open(file_name, 'r', encoding='utf-8')
    tfidf_kp = []
    corpus = []
    json_lines = json_file.readlines()
    for line in json_lines:
        json_data = json.loads(line)
        cur_content = json_data['title'].strip().lower() + ' ' + json_data['abstract'].strip().lower()
        corpus.append(cur_content)
    vectorizer = CountVectorizer(lowercase=True)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    assert len(weight) == len(json_lines)
    for line_index in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        weight_index = dict()
        for word_index in range(len(word)):
            weight_index[weight[line_index][word_index]] = word_index
        sorted_weight_index = dict(
            sorted(weight_index.items(), key=operator.itemgetter(0), reverse=True)[0: min(topk, len(word)): 1])
        keywords_list = []
        for keyword_weight in sorted_weight_index:
            keywords_list.append(word[sorted_weight_index[keyword_weight]])
            # print(word[sorted_weight_index[keyword_weight]], keyword_weight)
        kp_list = get_kp(corpus[line_index], keywords_list)
        tfidf_kp.append(kp_list)
    json_file.close()
    return tfidf_kp

def get_textRank_kp(file_name, topk):
    json_file = open(file_name, 'r', encoding='utf-8')
    textRank_kp = []
    for line in json_file.readlines():
        json_data = json.loads(line)
        cur_content = json_data['title'].strip().lower() + ' ' + json_data['abstract'].strip().lower()
        tr4w = TextRank4Keyword()
        tr4w.analyze(text=cur_content, lower=True, window=2)
        keywords_list = []
        for item in tr4w.get_keywords(topk, word_min_len=1):
            keywords_list.append(item.word)
        kp_list = get_kp(cur_content, keywords_list)
        # textRank_kp = tr4w.get_keyphrases(keywords_num=20, min_occur_num=2)
        textRank_kp.append(kp_list)
    json_file.close()
    return textRank_kp

def get_golden_kp(file_name):
    golden_kp = []
    json_file = open(file_name, 'r', encoding='utf-8')
    for line in json_file.readlines():
        json_data = json.loads(line)
        if file_name == 'data/ACL2017/kp20k/kp20k_train.json':
            golden_list = [kp.strip().lower() for kp in json_data['keywords']]
        else:
            golden_list = [kp.strip().lower() for kp in json_data['keywords'].split(';')]
        golden_kp.append(golden_list)
    json_file.close()
    return golden_kp

# def test_f1(kp, golden_kp, topk=5):
#     res={}
#     assert len(kp) == len(golden_kp)
#     true_num = 0.0
#     pre_num = 0.0
#     golden_num = 0.0
#     for line_index in range(len(kp)):
#         cur_pre_kp = kp[line_index][ : min(len(kp[line_index]), topk)]  # list
#         cur_golden_kp = golden_kp[line_index]            #list
#         pre_num += len(cur_pre_kp)
#         golden_num += len(cur_golden_kp)
#         for key in cur_pre_kp:
#             if key in cur_golden_kp:
#                 true_num += 1
#     res['p'] = true_num / pre_num * 100
#     res['r'] = true_num / golden_num *100
#     res['f1'] = 2 * res['p'] * res['r'] / (res['p'] + res['r'])
#     return res

def get_test_f1(pre_kp, golden_kp, topk=5):
    res={}
    assert len(pre_kp) == len(golden_kp)
    single_true_num = 0.0
    single_pre_num = 0.0
    single_golden_num = 0.0
    more_true_num = 0.0
    more_pre_num = 0.0
    more_golden_num = 0.0
    for line_index in range(len(pre_kp)):
        cur_single_pre_kp = []
        cur_more_pre_kp = []
        origin_pre_kp = pre_kp[line_index]
        numOfSingleKp = 0
        numOfMoreKp = 0
        booleanSingle = True
        booleanMore = True
        for kp in origin_pre_kp:
            if (booleanSingle and len(kp.split(' ')) == 1):
                single_pre_num += 1
                cur_single_pre_kp.append(kp)
                numOfSingleKp += 1
                if(numOfSingleKp >= min(len(origin_pre_kp), topk)):
                    booleanSingle = False
            elif (booleanMore and len(kp.split(' ')) > 1):
                more_pre_num += 1
                cur_more_pre_kp.append(kp)
                numOfMoreKp += 1
                if (numOfMoreKp >= min(len(origin_pre_kp), topk)):
                    booleanMore = False
            else:
                break

        cur_single_golden_kp = []
        cur_more_golden_kp = []
        origin_golden_kp = golden_kp[line_index]
        numOfSingleKp = 0
        numOfMoreKp = 0
        booleanSingle = True
        booleanMore = True
        for kp in origin_golden_kp:
            if (booleanSingle and len(kp.split(' ')) == 1):
                single_golden_num += 1
                cur_single_golden_kp.append(kp)
                numOfSingleKp += 1
                if (numOfSingleKp >= min(len(origin_golden_kp), topk)):
                    booleanSingle = False
            elif (booleanMore and len(kp.split(' ')) > 1):
                more_golden_num += 1
                cur_more_golden_kp.append(kp)
                numOfMoreKp += 1
                if (numOfMoreKp >= min(len(origin_golden_kp), topk)):
                    booleanMore = False
            else:
                break
        for kp in cur_single_pre_kp:
            if kp in cur_single_golden_kp:
                single_true_num += 1
        for kp in cur_more_pre_kp:
            if kp in cur_more_golden_kp:
                more_true_num += 1
    res['single_p'] = single_true_num / single_pre_num * 100
    res['single_r'] = single_true_num / single_golden_num * 100
    res['single_f1'] = 2 * res['single_p'] * res['single_r'] / (res['single_p'] + res['single_r'])
    if more_pre_num == 0:
        res['more_f1'] = 0
    else:
        res['more_p'] = more_true_num / more_pre_num * 100
        res['more_r'] = more_true_num / more_golden_num * 100
        res['more_f1'] = 2 * res['more_p'] * res['more_r'] / (res['more_p'] + res['more_r'])
    return res


def main():
    file_name = 'data/ACL2017/kp20k/kp20k_test.json'
    topk = 10
    tfidf_kp = get_tfidf_kp(file_name, 20)
    textRank_kp = get_textRank_kp(file_name, 20)
    rake_kp = get_rake_kp(file_name, 20)

    golden_kp = get_golden_kp(file_name)
    tfidf_res = get_test_f1(tfidf_kp, golden_kp, topk=topk)
    textRank_res = get_test_f1(textRank_kp, golden_kp, topk=topk)
    rake_res = get_test_f1(rake_kp, golden_kp, topk=topk)
    print(file_name)
    print('topk = {}'.format(topk))
    print("tfidf_res" + str(tfidf_res))
    print("textRank_res" + str(textRank_res))
    print("rake_res" + str(rake_res))
    # print('tf_idf Precision :{:.2f}, Recall :{:.2f}, F1 score : {:.2f}'.format(tfidf_res['p'], tfidf_res['r'], tfidf_res['f1']))
    # print('textRank Precision :{:.2f}, Recall :{:.2f}, F1 score : {:.2f}'.format(textRank_res['p'], textRank_res['r'],textRank_res['f1']))
    # print('RAKE Precision :{:.2f}, Recall :{:.2f}, F1 score : {:.2f}'.format(rake_res['p'], rake_res['r'],rake_res['f1']))

    # print(golden_kp[9])
    # print(rake_kp[9])
    # i = 0
    # for line in rake_kp:
    #     i += 1
    #     if len(line) > 10:
    #         print(i)
    #         break

if __name__ == '__main__':
    main()
