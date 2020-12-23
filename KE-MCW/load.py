# -*- coding: utf-8 -*-
import numpy as np
import pickle
import random
def atisfold_old():
    f_data_set = open('data/data_set.pkl', 'rb')
    f_emb = open('data/embedding.pkl', 'rb')
    f_idx2word = open('data/idx2words.pkl', 'rb')
    f_char_emb = open('data/char_embedding.pkl', 'rb')
    f_char2idx = open('data/char2idx.pkl', 'rb')
    train_set, test_set, dicts = pickle.load(f_data_set)
    embedding = pickle.load(f_emb)
    idx2word = pickle.load(f_idx2word)
    char_emb = pickle.load(f_char_emb)
    char2idx = pickle.load(f_char2idx)
    f_data_set.close()
    f_emb.close()
    f_idx2word.close()
    f_char_emb.close()
    f_char2idx.close()
    return train_set, test_set, dicts, embedding, idx2word, char_emb, char2idx

def atisfold(data_set_file, emb_file):
    f_data_set = open(data_set_file, 'rb')
    f_emb = open(emb_file, 'rb')
    train_set, test_set, dicts = pickle.load(f_data_set)
    embedding = pickle.load(f_emb)
    f_data_set.close()
    f_emb.close()
    return train_set, test_set, dicts, embedding

def atisfold_ACL2017(data_set_file, emb_file):
    f_data_set = open(data_set_file, 'rb')
    f_emb = open(emb_file, 'rb')
    train_set, valid_set, test_set, dicts = pickle.load(f_data_set)
    embedding = pickle.load(f_emb)
    f_data_set.close()
    f_emb.close()
    return train_set, valid_set, test_set, dicts, embedding

def pad_sentences(sentences, padding_word=0, forced_sequence_length=None):
    if forced_sequence_length is None:
        sequence_length=max(len(x) for x in sentences)
    else:
        sequence_length=forced_sequence_length
    padded_sentences=[]
    for i in range(len(sentences)):
        sentence=sentences[i]
        num_padding=sequence_length-len(sentence)
        if num_padding<0:
            padded_sentence=sentence[0:sequence_length]
        else:
            padded_sentence=sentence+[int(padding_word)]*num_padding

        padded_sentences.append(padded_sentence)

    return padded_sentences

def pad_chars(char_input_x, padding=0, forced_word_length=20, forced_sentence_length=None):
    if forced_sentence_length is None:
        sentence_length=max(len(sentence) for sentence in char_input_x)
    else:
        sentence_length=forced_word_length
    word_length = forced_word_length
    res_char_input_x = []
    for num_sent in range(len(char_input_x)):
        padded_words = []
        cur_sent = char_input_x[num_sent]
        for num_word in range(len(cur_sent)):
            word = cur_sent[num_word]
            num_padding = word_length - len(word)
            if num_padding<0:
                padded_word=word[0:word_length]
            else:
                padded_word=word+[int(padding)]*num_padding
            padded_words.append(padded_word)
        sentence_pad_num = sentence_length - len(padded_words)
        if sentence_pad_num > 0:
            for i in range(sentence_pad_num):
                padded_words.append([int(padding)]*forced_word_length)
        else:
            padded_words = padded_words[0 : sentence_length]
        res_char_input_x.append(padded_words)
    return res_char_input_x








