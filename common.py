import numpy as np
import logging
from embedding import vocab_file_path, tokenize
import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score

np.random.seed(0)

handler = logging.FileHandler('./log/train.log', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

SENTENCE_LENGTH = 100
unseen = '@UNSEEN@'

dictionary = json.load(open('%s.json' % (vocab_file_path)))

batch_size = 2000

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'normal']

def get_numbered_label(data, categories):
    label = pd.Series([0]*data.shape[0])
    for i, category in enumerate(categories):
        label[(data[category] == 1).values] = i+1
    return label.values

def padding(batch, batch_size, feat_dim):
    batch = np.array(batch)
    zeros = np.zeros((max(batch_size, batch.shape[0]), feat_dim))
    for i in range(batch.shape[0]):
        batch[i] = batch[i][:min(SENTENCE_LENGTH, len(batch[i]))]
        zeros[i, :len(batch[i])] = batch[i]
    batch = zeros
    return batch

def to_word_id(batch):
    '''
    for sentence in batch:
        for tok in tokenize(sentence):
            if tok in dictionary:
                print(dictionary[tok.lower()], tok)
            else:
                print('none %s'%(tok))
    '''
    batch = [[dictionary[tok] if tok in dictionary else dictionary[unseen] for tok in tokenize(sentence)] for sentence in batch]
    return batch

def preprocess(batch, batch_size):
    labels = padding(batch[categories], batch_size, len(categories))
    comments_text = batch['comment_text']
    comments_ids = to_word_id(comments_text)
    comments = padding(comments_ids, batch_size, SENTENCE_LENGTH)
    data = np.concatenate((comments, labels), axis=1)
    return data

class Batch:
    def __init__(self, df, preprocess, batch_size=100):
        self.df = df.sample(frac=1)
        self.index = 0
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.data = self.preprocess(self.df, self.df.shape[0])

    def __next__(self):
        try:
            if self.index >= self.df.shape[0]:
                raise IndexError

            next_index = self.index+self.batch_size
            if next_index > self.df.shape[0]:
                raise IndexError
                #next_index = self.df.shape[0]

            #self.batch = self.df.iloc[self.index:next_index, :]
            data, label = self.data[self.index:next_index, :SENTENCE_LENGTH], self.data[self.index:next_index, SENTENCE_LENGTH:]
            self.index = next_index
            return data, label
        except IndexError:
            self.index = 0
            self.data = np.random.permutation(self.data)
            raise StopIteration

    def __iter__(self):
        return self

def to_pred(predicts):
    pred = (predicts >= 0.5).astype(int)
    return pred

def evaluate(predicts, labels):
    pred = to_pred(predicts)
    # skip UndefinedMetricWarning when al lprediction is 0/1
    pred[0] = 1

    precision, recall = precision_score(labels, pred), recall_score(labels, pred)
    return precision, recall

def test_embed():
    import tensorflow as tf
    vocab_shape = (40305, 300)
    with tf.Session() as sess:
        W = tf.get_variable(name='W', shape=vocab_shape, trainable=True)
        saver = tf.train.Saver(var_list=[W])
        saver.restore(sess, './models/embed_matrix.ckpt')

        x = tf.placeholder(shape=(None, None), dtype=tf.int32)
        embed = tf.nn.embedding_lookup(W, x)
        df = pd.read_csv('data/train_over_sampled.csv').sample(frac=1)
        bt = Batch(df, preprocess, batch_size=1000)
        for batch in bt:
            e  = sess.run([embed], feed_dict={x: batch[0]})

def test_batch():
    df = pd.read_csv('data/train_over_sampled.csv').sample(n=100)
    bt = Batch(df, preprocess, batch_size=10)
    for batch in bt:
        b, l = batch
        print(l.tolist())
    print(bt.index)
    for batch in bt:
        b, l = batch
        print(l.tolist())

if __name__ == '__main__':
    #test_embed()
    test_batch()
