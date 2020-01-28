import re
import operator


def get_test_truth(my_ipc):
    truth = dict()
    for cluster_label in my_ipc:
        cur_result_count = dict()
        cur_label_ipcs = my_ipc[cluster_label]
        for cur_ipc in cur_label_ipcs:
            if cur_ipc not in cur_result_count:
                cur_result_count[cur_ipc] = 1
            else:
                cur_result_count[cur_ipc] += 1
        cur_result_count = dict(sorted(cur_result_count.items(), key=operator.itemgetter(1), reverse=True))
        predict_ipc = list(cur_result_count.keys())[0]
        truth[cluster_label] = predict_ipc
    return truth

if __name__ == '__main__':
    cluster_result_name = r'D:\PycharmProjects\KeywordExtraction\data\patent_abstract\Birch\bxk_abstract_wordAVG_keywordTest_0.78_50.txt'
    my_ipc = dict()
    ipc_num = 0
    with open(cluster_result_name, 'r', encoding='utf-8') as result_f:
        result_lines = result_f.readlines()
        line_num = 0
        if_write = False
        cur_label = -1
        while line_num < len(result_lines):
            search_title = re.search('类标签为:', result_lines[line_num])
            if search_title:
                cur_label = int(result_lines[line_num].split(':')[1])
                if_write = True
                line_num += 2
            if if_write:
                cur_label_ipc = result_lines[line_num].split(':  ')[1].split('   ')[0].strip()
                if cur_label not in my_ipc:
                    my_ipc[cur_label] = [cur_label_ipc]
                    ipc_num += 1
                else:
                    my_ipc[cur_label].append(cur_label_ipc)
                    ipc_num += 1
                line_num += 1
            else:
                line_num += 1
    truth = get_test_truth(my_ipc)
    print('预测的类标签为：')
    print(truth)
    error = 0.0
    for label in truth:
        for label_ipc in my_ipc[label]:
            if label_ipc != truth[label]:
                error += 1
    print('聚类准确率为：%f%%' % (100-error/ipc_num*100))
