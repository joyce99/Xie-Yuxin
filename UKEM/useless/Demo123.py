import numpy as np
def read(file, threshold=0, dtype='float'):
    print('读取词向量文件中......')
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype)
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        words.append(word)
        matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
    return (words, matrix)

embedding_name = 'test.vector'
embedding_file = open(embedding_name, 'r', encoding='utf-8', errors='surrogateescape')
words, wordvecs = read(embedding_file)
word2ind = {word: i for i, word in enumerate(words)}