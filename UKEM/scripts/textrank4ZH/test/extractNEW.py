import pymysql
import sys
import io
import codecs
from textrank4zh import TextRank4Keyword


sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码
if __name__ == '__main__':
    # db = pymysql.connect("localhost", "root", "", "patent_system")
    # # 使用 cursor() 方法创建一个游标对象 cursor
    # cursor = db.cursor()
    # patent_id = 0
    # sql = """ SELECT count(id) FROM tb_patent; """
    # try:
    #     # 执行sql语句
    #     cursor.execute(sql)
    #     # 提交到数据库执行
    #     # 获取所有记录列表
    #     results = cursor.fetchall()
    #     patent_num = results[0][0]
    #     print('专利数目：' + str(patent_num) )
    #     db.commit()
    # except IndexError as e:
    #     # 如果发生错误则回滚
    #     db.rollback()
    #     print(e)
    # with open('TextRank_test.txt', 'a') as file_log:
    #     file_log.write('%d\t%s\n' % (patent_id + 1, xml_name))

    log_file = open(r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\test\bxd_textRank_abstract.txt', 'w', encoding='utf-8')
    # sql = """ SELECT abstract FROM tb_patent;  """
    # try:
        # 执行sql语句
        # cursor.execute(sql)
        # 提交到数据库执行
        # 获取所有记录列表
        # results = cursor.fetchall()
    with open(r'D:\PycharmProjects\Dataset\keywordEX\patent\bxd\_bxd_label_abstract.txt', 'r', encoding='utf-8') as test_file:
        num = 0
        for row in test_file.readlines():
            line_split = row.split(' ::  ')
            if len(line_split) == 2:
                content = line_split[1].strip()
                print('第%d条专利摘要：' % (num + 1))
                print(content)
                log_file.write('第%d条专利摘要：\t\t%s\n' % (num + 1, line_split[0]))
                log_file.write('%s\n' % content)
                log_file.write('-------keyword-------\n')
                tr4w = TextRank4Keyword(stop_words_file='D:\PycharmProjects\KeywordExtraction\data\patent_abstract\TextRankstop.txt')
                tr4w.analyze(text=content, lower=True, window=3, vertex_source = 'words_no_stop_words', pagerank_config={'alpha': 0.85})
                for item in tr4w.get_keywords(20, word_min_len=2):
                    log_file.write('%s\t\t%f\n' % (item.word, item.weight))
                num += 1
                log_file.write('----------------------------------------------------------------\n')
            # print(row[0] + "\n")
            # log_file.write('%s\n\n' % (row[0]))
            # log_file.write('-------keyword-------\n')
            # tr4w = TextRank4Keyword(stop_words_file = '../textrank4zh/stopwords.txt')
            # tr4w.analyze(text=row[0], lower=True, vertex_source = 'no_stop_words', window=3, pagerank_config={'alpha': 0.85})
            # for item in tr4w.get_keywords(20, word_min_len=2):
            #     log_file.write('%s\t%f\n' % (item.word, item.weight))
            # log_file.write("-------phrase-------\n")
            # for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num=0):
            #     log_file.write('%s\n' % (phrase))
            # i+=1
            # print("第%d条专利成功写入文档！\n" % (i))
            # log_file.write('\n')
        # db.commit()
    # except IndexError as e:
    #     # 如果发生错误则回滚
    #     db.rollback()
    #     print(e)
    #for num in range(1, patent_num+1):


    # 关闭数据库连接
    # db.close()
    log_file.close()
