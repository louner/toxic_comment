import json
import numpy as np
import tensorflow as tf
import pandas as pd
from nltk import word_tokenize
embedding_size = 300

from gensim.models.keyedvectors import KeyedVectors
word2vec_model_filepath = '/home/louner/school/ml/word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin'

from collections import Counter

vocab_file_path = './data/vocab'
batch_size = 5
LEAST_WORD_COUNT = 5

import logging

dictionary = json.load(open('%s.json' % (vocab_file_path)))

def tokenize(sentence):
    for tok in word_tokenize(sentence):
        yield tok

def preprocess(sentences):
    sentences_preprocessed = []
    for sentence in sentences:
        if type(sentence) != str:
            continue

        if sentence[-1] == '?':
            sentence = sentence[:-1]
        sentences_preprocessed.append(sentence.lower())
    return sentences_preprocessed

def make_vocab():
    df = pd.read_csv('data/train.csv')
    sentences = df['comment_text'].values.tolist()

    df = pd.read_csv('data/test.csv')
    sentences += df['comment_text'].values.tolist()

    sentences = preprocess(sentences)

    vocabs = Counter([tok for sentence in sentences for tok in tokenize(sentence)])
    with open('./data/vocab.json', 'w') as f:
        json.dump(vocabs, f)

def make_embed_matrix(vocab_file_path):
    handler = logging.FileHandler('./log/embeddding.log', mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    w2v = KeyedVectors.load_word2vec_format(word2vec_model_filepath, binary=True)
    dictionary = {'@UNSEEN@': 0}
    embed_matrix = [[0]*embedding_size]

    with open('data/vocab.json') as f:
        vocab = json.load(f)

    absent_words = []
    for word, count in vocab.elemnts():
        if count > LEAST_WORD_COUNT:
            try:
                vec = w2v.word_vec(word)
                embed_matrix.append(vec)
                dictionary[word] = len(dictionary)
            except:
                absent_words.append(word)
                logger.error('UNKNOWN %s'%(word))


    embed_matrix = np.array(embed_matrix)
    logger.info('shape: %s'%str(embed_matrix.shape))

    json.dump(dictionary, open('%s.json'%(vocab_file_path), 'w'))

    return np.reshape(embed_matrix, embed_matrix.shape)

def save_embed():
    embed_matrix = make_embed_matrix(vocab_file_path=vocab_file_path)

    W = tf.get_variable(name='W', shape=embed_matrix.shape, initializer=tf.constant_initializer(embed_matrix))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(var_list=[W])
        saver.save(sess, './models/embed_matrix.ckpt')

def load_embed():
    W = tf.get_variable(name='W', shape=shape, trainable=False)
    ids = tf.constant([0, 1, 0])
    lookup = tf.nn.embedding_lookup(params=W, ids=ids)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver(var_list=[W])
        saver.restore(sess, './models/embed_matrix.ckpt')
        print(sess.run(lookup))

def to_ids(sentence):
    toks = sentence.decode('utf-8').split(' ')
    ids = [dictionary[tok] for tok in toks if tok in dictionary]
    return np.array(ids, dtype=np.int32)

def to_sentence_matrix(sentence):
    ids = tf.py_func(to_ids, [sentence], tf.int32)
    embedding = tf.nn.embedding_lookup(params=W, ids=ids)
    return embedding

def trans():
    W = tf.get_variable(name='W', shape=shape, trainable=False)

    sentence1 = 'What is the step by step guide to invest in share market in india?'
    sentence2 = 'What is the step by step guide to invest in share market?'
    sentence_matrix1, sentence_matrix2 = to_sentence_matrix(sentence1), to_sentence_matrix(sentence2)

    with tf.Session() as sess:
        saver = tf.train.Saver(var_list=[W])
        saver.restore(sess, './models/embed_matrix.ckpt')

        print(sess.run([sentence_matrix1, sentence_matrix2]))

