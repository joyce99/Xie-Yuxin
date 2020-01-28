import pymysql
import jieba
import sys
import io
import codecs
import re
import csv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8') #改变标准输出的默认编码
if __name__ == '__main__':
    db = pymysql.connect("localhost", "root", "", "patent_system")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    # log_file = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\ydy\_0bx1dh_abstract.csv', 'w', newline='', encoding='utf-8-sig')
    log_file = open(r'D:\PycharmProjects\Dataset\keywordEX\patent\bxk\_bxk_label_abstract.txt', 'w', encoding='utf-8')
    # f_csv = csv.writer(log_file)
    # headers = ['label', 'abstract']
    # headers = ['label', 'tech_field']
    # f_csv.writerow(headers)
    sql = """ SELECT label, abstract FROM tb_patentall_label where (label LIKE '%F25D%') OR (label LIKE '%D06F%') OR (label LIKE '%F24F%'); """
    # sql = """ SELECT label,abstract FROM tb_patentall_label where label LIKE '%D06F%'; """
    num0 = 0
    num1 = 0
    num2 = 0
    try:
        # 执行sql语句
        cursor.execute(sql)
        # 提交到数据库执行
        # 获取所有记录列表
        results = cursor.fetchall()
        i = 0
        for row in results:
            if re.search('F25D', row[0]) and num0 < 1000:
                num0 += 1
                # print(row[0] + ' ::  ' + row[1])
                log_file.write('%s ::  %s\n' % (row[0], row[1]))
                # f_csv.writerow([row[0], row[1]])
                print(row[1])
                # log_file.write('%s\n' % row[1])
                i += 1
                print("第%d条专利成功写入文档!" % i)
                print('************************************************************************')
            if re.search('D06F', row[0]) and num1 < 1000:
                num1 += 1
                # print(row[0] + ' ::  ' + row[1])
                log_file.write('%s ::  %s\n' % (row[0], row[1]))
                print(row[1])
                # log_file.write('%s\n' % row[1])
                # f_csv.writerow([row[0], row[1]])
                i += 1
                print("第%d条专利成功写入文档!" % i)
                print('************************************************************************')
            if re.search('F24F', row[0]) and num2 < 1000:
                num2 += 1
                # print(row[0] + ' ::  ' + row[1])
                log_file.write('%s ::  %s\n' % (row[0], row[1]))
                print(row[1])
                # log_file.write('%s\n' % row[1])
                # f_csv.writerow([row[0], row[1]])
                i += 1
                print("第%d条专利成功写入文档!" % i)
                print('************************************************************************')

        # for row in results:      #YDY
        #     if num0 == 1000:
        #         break
        #     else:
        #         if re.search('D06F', row[0]):
        #             # cur_label = 0
        #             print(row[0] + ' ::  ' + row[1])
        #             # word_list = jieba.cut(row[1])
        #             # cur_abstract = ' / '.join(word_list)
        #             # log_file.write('%d ::  %s\n' % (i+1, cur_abstract))
        #             log_file.write('%d ::  %s\n' % (i+1, row[1]))
        #             # log_file.write('keywords:\n\n')
        #             # f_csv.writerow([cur_label, row[1]])
        #             num0 +=1
        #             i += 1
        #         print("第%d条专利成功写入文档!" % i)
        #         print('************************************************************************')

        # for row in results:
        #     if num1 == 500:
        #         break
        #     else:
        #         print(row[0] + ' ::  ' + row[1])
        #         if re.search('H04M', row[0]):
        #             cur_label = 1
        #             # log_file.write('%s ::  %s\n' % (cur_label, row[1]))
        #             f_csv.writerow([cur_label, row[1]])
        #             num1 += 1
        #             i += 1
        db.commit()
    except IndexError as e:
        # 如果发生错误则回滚
        db.rollback()
        print(e)



    # 关闭数据库连接
    db.close()
    log_file.close()
    print('num0:' + str(num0))
    print('num1:' + str(num1))
    print('num2:' + str(num2))
