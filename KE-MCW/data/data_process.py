import numpy as np
import re
import pickle
from collections import Counter
import gensim
import random


def getlist(filename):
    with open(filename,'r',encoding='utf-8') as f:
        datalist,taglist=[],[]
        for line in f:
            line=line.strip()
            datalist.append(line.split('\t')[0])
            taglist.append(line.split('\t')[1])
    return datalist,taglist

#build vocabulary
def get_dict(filenames):
    trnTweet,testTweet=filenames
    sentence_list=getlist(trnTweet)[0]+getlist(testTweet)[0]
    words=[]
    for sentence in sentence_list:
        word_list=sentence.split()
        words.extend(word_list)
    word_counts=Counter(words)
    words2idx={word[0]:i+1 for i,word in enumerate(word_counts.most_common())}
    idx2words = {v: k for (k,v) in words2idx.items()}
    labels2idx = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
    dicts = {'words2idx': words2idx, 'labels2idx': labels2idx, 'idx2words': idx2words}

    return dicts

def get_train_test_dicts(filenames):
    """
    Args:
    filenames:trnTweet,testTweet,tag_id_cnt

    Returns:
    dataset:train_set,test_set,dicts

    train_set=[train_lex,train_y,train_z]
    test_set=[test_lex,test_y,test_z]
    dicts = {'words2idx': words2idx, 'labels2idx': labels2idx}


    """
    trnTweetCnn, testTweetCnn= filenames
    dicts=get_dict([trnTweetCnn,testTweetCnn])

    trn_data=getlist(trnTweetCnn)
    test_data=getlist(testTweetCnn)

    trn_sentence_list,trn_tag_list=trn_data
    test_sentence_list,test_tag_list=test_data
    
    words2idx=dicts['words2idx']
    labels2idx=dicts['labels2idx']

    def get_lex_y(sentence_list,tag_list,words2idx):
        lex,y,z=[],[],[]
        bad_cnt=0
        for s,tag in zip(sentence_list,tag_list):
       
            

            word_list=s.split()
            t_list=tag.split()

            emb=list(map(lambda x:words2idx[x],word_list))


            begin=-1
            for i in range(len(word_list)):
                ok=True
                for j in range(len(t_list)):
                    if word_list[i+j]!=t_list[j]:
                        ok=False;
                        break
                if ok==True:
                    begin=i
                    break

            if begin==-1:
                bad_cnt+=1
                continue

            lex.append(emb)

            labels_y=[0]*len(word_list)
            for i in range(len(t_list)):
                labels_y[begin+i]=1
            y.append(labels_y)

            labels_z=[0]*len(word_list)
            if len(t_list)==1:
                labels_z[begin]=labels2idx['S']
            elif len(t_list)>1:
                labels_z[begin]=labels2idx['B']

                for i in range(len(t_list)-2):
                    labels_z[begin+i+1]=labels2idx['I']
                labels_z[begin+len(t_list)-1]=labels2idx['E']

            z.append(labels_z)
        return lex,y,z
    
    train_lex, train_y, train_z = get_lex_y(trn_sentence_list,trn_tag_list, words2idx)  # train_lex: [[每条tweet的word的idx],[每条tweet的word的idx]], train_y: [[关键词的位置为1]], train_z: [[关键词的位置为0~4(开头、结尾...)]]
    test_lex, test_y, test_z = get_lex_y(test_sentence_list,test_tag_list,words2idx)
    train_set = [train_lex, train_y, train_z]
    test_set = [test_lex, test_y, test_z]
    data_set = [train_set, test_set, dicts]
    with open('../CNTN/data/inspec_wo_stem/data_set.pkl', 'wb') as f:
        pickle.dump(data_set, f)
        # dill.dump(data_set, f)
    return data_set


def get_CNTN_train_test_dicts(filenames):
    """
    Args:
    filenames:trnTweet,testTweet,tag_id_cnt

    Returns:
    dataset:train_set,test_set,dicts

    train_set=[train_lex,train_y,train_z]
    test_set=[test_lex,test_y,test_z]
    dicts = {'words2idx': words2idx, 'labels2idx': labels2idx}


    """
    trnTweetCnn, testTweetCnn = filenames
    dicts = get_dict([trnTweetCnn, testTweetCnn])

    trn_data = getlist(trnTweetCnn)
    test_data = getlist(testTweetCnn)

    trn_sentence_list, trn_tag_list = trn_data
    test_sentence_list, test_tag_list = test_data

    words2idx = dicts['words2idx']
    labels2idx = dicts['labels2idx']

    def get_CNTN_lex_y(sentence_list, tag_list, words2idx):
        lex, y, z = [], [], []
        for s, tag in zip(sentence_list, tag_list):
            word_list = s.split()
            t_list = tag.split()
            emb = list(map(lambda x: words2idx[x], word_list))
            i = 0
            find_keyphrase = False
            len_keyphrase = 0
            all_keyphrase_sub = []
            cur_keyphrase_sub = []
            while i < len(word_list):
                cur_word = word_list[i]
                j = 0
                while j < len(t_list):
                    cur_keyword = t_list[j]
                    if cur_word == cur_keyword:
                        len_keyphrase += 1
                        cur_keyphrase_sub.append(i)
                        find_keyphrase = True
                        j = 0
                        # i += 1
                        break
                    elif find_keyphrase and j == len(t_list)-1:
                        all_keyphrase_sub.append(cur_keyphrase_sub)
                        cur_keyphrase_sub = []
                        find_keyphrase = False
                        j += 1
                        # i += 1
                    else:
                        # tag_again = False
                        j += 1
                        # if j == len(t_list):
                        #     i += 1
                        continue
                i += 1

            lex.append(emb)
            cur_y = [ 0 for k in range(len(word_list))]
            cur_z = [ 0 for k in range(len(word_list))]
            for cur_sub in all_keyphrase_sub:
                if len(cur_sub) == 1:
                    cur_y[cur_sub[0]] = 1
                    cur_z[cur_sub[0]] = labels2idx['S']
                elif len(cur_sub) > 1:
                    cur_y[cur_sub[0]] = 1
                    cur_z[cur_sub[0]] = labels2idx['B']
                    for k in range(len(cur_sub) - 2):
                        cur_y[cur_sub[1+k]] = 1
                        cur_z[cur_sub[1+k]] = labels2idx['I']
                    cur_y[cur_sub[-1]] = 1
                    cur_z[cur_sub[-1]] = labels2idx['E']

            y.append(cur_y)
            z.append(cur_z)
        return lex, y, z

    train_lex, train_y, train_z = get_CNTN_lex_y(trn_sentence_list, trn_tag_list,
                                            words2idx)  # train_lex: [[每条tweet的word的idx],[每条tweet的word的idx]], train_y: [[关键词的位置为1]], train_z: [[关键词的位置为0~4(开头、结尾...)]]
    test_lex, test_y, test_z = get_CNTN_lex_y(test_sentence_list, test_tag_list, words2idx)
    train_set = [train_lex, train_y, train_z]
    test_set = [test_lex, test_y, test_z]
    data_set = [train_set, test_set, dicts]
    with open('../CNTN/data/semeval_wo_stem/data_set123.pkl', 'wb') as f:
        pickle.dump(data_set, f)
        # dill.dump(data_set, f)
    return data_set

def load_bin_vec(frame,vocab):
    k = 0
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(frame, binary=True)
    vec_vocab = model.vocab
    for word in vec_vocab:
        embedding = model[word]
        if word in vocab:
            word_vecs[word] = np.asarray(embedding,dtype=np.float32)
        k += 1
        if k % 10000 == 0:
            print("load_bin_vec %d" % k)
    return word_vecs

# def load_txt_vec(frame,vocab):
#     k=0
#     word_vecs={}
#     with open(frame,'r',encoding='utf-8') as f:
#         for line in f.readlines():
#             word=line.strip().split('\t',1)[0]
#             embeding=line.strip().split('\t',1)[1].split()
#             if word in vocab:
#                 word_vecs[word]=np.asarray(embeding,dtype=np.float32)
#             k+=1
#             if k%10000==0:
#                 print ("load_bin_vec %d" % k)
#
#     return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, dim=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    k=0
    for w in vocab:
        if w not in word_vecs:
            word_vecs[w]=np.asarray(np.random.uniform(-0.25,0.25,dim),dtype=np.float32)
            k+=1
            if k % 10000==0:
                print ("add_unknow_words %d" % k)
    return word_vecs

def get_embedding(w2v,words2idx,k=300):
    embedding = np.zeros((len(w2v) + 2, k), dtype=np.float32)
    for (w,idx) in words2idx.items():
        embedding[idx]=w2v[w]
    #embedding[0]=np.asarray(np.random.uniform(-0.25,0.25,k),dtype=np.float32)
    with open('../CNTN/data/semveal_wo_stem/embedding.pkl','wb') as f:
        pickle.dump(embedding,f)
    return embedding


if __name__ == '__main__':
    data_folder = ["../CNTN/data/semeval_wo_stem/mytrain123.txt","../CNTN/data/semeval_wo_stem/mytestNEW.txt"]
    data_set = get_CNTN_train_test_dicts(data_folder)
    print ("data_set complete!")
    dicts = data_set[2]
    vocab = set(dicts['words2idx'].keys())
    print ("total num words: " + str(len(vocab)))
    print ("dataset created!")
    train_set, test_set, dicts=data_set
    print ("total train lines: " + str(len(train_set[0])))
    print("total test lines: " + str(len(test_set[0])))

    #GoogleNews-vectors-negative300.txt为预先训练的词向量
    #w2v_file = 'D:\PycharmProjects\myCNN_RNN_attention\data\original_data\GoogleNews-vectors-negative300.bin'
    #w2v = load_bin_vec(w2v_file,vocab)
    #print ("word2vec loaded")
    #w2v = add_unknown_words(w2v, vocab)
    #embedding=get_embedding(w2v,dicts['words2idx'])
    #print ("embedding created")

    # f = open("../CNTN/data/semeval_wo_stem/mytest.txt", 'r', encoding='utf-8')
    # w = open("../CNTN/data/semeval_wo_stem/mytestNEW.txt", 'w', encoding='utf-8')
    # lines = f.readlines()
    # for line in lines:
    #     content = line.split('\t')[0]
    #     keys = line.split('\t')[1]
    #     mycon1 = content[:(len(content)-1)//2]
    #     mycon2 = content[(len(content)-1)//2+1:]
    #     if mycon1 == mycon2:
    #         w.write(mycon1 + '\t' + keys)
    #     else:
    #         w.write(line)
    #     # print(len(content))
    # f.close()
    # w.close()
