from collections import Counter, deque
import nltk

def load_data(articles, token='_'):
    file_w = open('./data/inspec_wo_stem/mytrain.txt', 'w', encoding='utf-8')
    kw = deque()
    history = ''
    first_0 = True
    for article in articles:
        _ = article.strip().split(token)
        if _[2] == '1':
            first_0 = True
            doc = nltk.word_tokenize(_[0])
            cur_str = ' '.join(doc)
            if cur_str == history:
                kw.append(_[1])
            else:
                kw.clear()
                history = cur_str
                file_w.write(history + '\t')
                kw.append(_[1])
        elif first_0:
            cur_kws = ' '.join(kw)
            file_w.write(cur_kws + '\n')
            first_0 = False

    file_w.close()

def main():
    file_r = open('./data/inspec_wo_stem/train.txt', 'r', encoding='utf-8')
    sentences = file_r.readlines()
    load_data(sentences)
    file_r.close()


if __name__ == '__main__':
    main()