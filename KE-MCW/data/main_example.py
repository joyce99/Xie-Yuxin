# -*- coding: utf-8 -*-
import pickle
def main():
    
    f = open('data_set.pkl','rb')
    train_set, test_set, dicts = pickle.load(f)
    embedding = pickle.load(open('embedding.pkl','rb'))
  
    word2idx=dicts['words2idx']
    labels2idx=dicts['labels2idx']

    train_lex, train_y, train_z = train_set
    test_lex,  test_y, test_z  = test_set
    # 将测试集分成测试集和验证,最终训练集,验证集,测试集7:1:2
    tr = int(len(test_lex)*0.67)
    valid_lex, valid_y, valid_z = test_lex[tr:], test_y[tr:], test_z[tr:]
    test_lex,  test_y, test_z  = test_lex[:tr],test_y[:tr],test_z[:tr]


    print ('len(train_data) {}'.format(len(train_lex)))
    print ('len(valid_data) {}'.format(len(valid_lex)))
    print ('len(test_data) {}'.format(len(test_lex)))

    vocab_size = len(word2idx)
    print ('len(vocab) {}'.format(vocab_size))
    print ("Train started!")

main()