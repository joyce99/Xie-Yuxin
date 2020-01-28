import os

class myfile:
    def __init__(self):
        self.name = ""


def search(folder, filters, allfile):
    folders = os.listdir(folder)
    for name in folders:
        curname = os.path.join(folder, name)
        isfile = os.path.isfile(curname)
        if isfile:
            for filter in filters:
                if name.startswith(filter):
                    cur = myfile()
                    cur.name = curname
                    allfile.append(cur.name)
                    break
        else:
            search(curname, filters, allfile)
    return allfile


if __name__ == '__main__':
    folder = r"/Users/mac/Documents/LearnPython/KeywordExtraction/data/SemEval2010/train"
    # filters = ['C','H','I','J']
    filters = ['J']
    allfile = []
    allfile = search(folder, filters, allfile)
    file_len = len(allfile)
    print('共查找到%d个摘要文件' %(file_len))
    train_file = open('../data/SemEval2010_J_train.txt', 'a', encoding='utf-8')
    for f in allfile:
        with open(f, 'r', encoding='utf-8') as curf:
            for line in curf.readlines():
                train_file.write(line)

    train_file.close()






