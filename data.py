EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

FILENAME = 'data/reddit_data_set_short_words_1.27_milion.csv'

limit = {
        'maxq' : 21,
        'minq' : 1,
        'maxa' : 21,
        'mina' : 1
        }

UNK = 'unk'
VOCAB_SIZE = 6000

import random
import sys

import nltk
import itertools
from collections import defaultdict

import numpy as np

import pickle
import pandas


def ddefault():
    return 1



'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist




'''
 create the final dataset : 
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )
 
'''
def zero_pad(qtokenized, atokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32) 
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

def remove_columns(dataframe):
    dataframe = dataframe.drop(['subreddit', 'Unnamed: 0','score'], axis=1)
    dataframe[['reply_text','comment_text']] = dataframe[['reply_text','comment_text']].astype(str)
    return dataframe

def lower_case_word(df):
    # Capitalizes all the column headers
    df['reply_text'] = df['reply_text'].str.lower()
    df['comment_text'] = df['comment_text'].str.lower()
    return df

def split_row_words_to_list(df):
    df['reply_text'] = df.reply_text.str.replace(',' , '')
    df['comment_text'] = df.comment_text.str.replace(',' , '')
    df['reply_text_vec'] = df['reply_text'].str.split(' ')
    df['comment_text_vec'] = df['comment_text'].str.split(' ')
    return df

def tokenized(df):
    return df['comment_text_vec'].values,df['reply_text_vec'].values

def process_data(PATH):

    print('\n>> Run Panda pipline')

    df = pandas.read_csv(PATH+FILENAME, sep='\t',dtype={"reply_text": str, "comment_text": str,"subreddit":str})

    qtokenized,atokenized = (df.pipe(remove_columns)
        .pipe(lower_case_word)
        .pipe(split_row_words_to_list)
        .pipe(tokenized)
    )

    print(qtokenized[60],atokenized[60])

    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx)

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save('datasets/idx_q.npy', idx_q)
    np.save('datasets/idx_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open('datasets/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

def load_data(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a

def load_metadata(PATH=''):
    # read data control dictionaries
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return metadata



if __name__ == '__main__':
    process_data()
