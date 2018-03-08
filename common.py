import numpy as np
import logging
from embedding import vocab_file_path, tokenize
import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler, SMOTE

np.random.seed(0)

def init_logger(logger_path):
    handler = logging.FileHandler(logger_path, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger = logging.getLogger(logger_path)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

SENTENCE_LENGTH = 200
CHAR_LENGTH = 32
unseen = '@UNSEEN@'

dictionary = json.load(open('%s.json' % (vocab_file_path)))

#categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'normal']
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def get_numbered_label(data, categories):
    label = pd.Series([0]*data.shape[0])
    for i, category in enumerate(categories):
        label[(data[category] == 1).values] = i+1
    return label.values

def padding(batch, batch_size, feat_dim):
    batch = np.array(batch)
    #zeros = np.zeros((max(batch_size, batch.shape[0]), feat_dim))
    zeros = np.zeros((batch_size, feat_dim))
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
    # remove unseen
    batch = [[dictionary[tok] if tok in dictionary else dictionary[unseen] for tok in tokenize(sentence) ] for sentence in batch]
    return batch

def preprocess(batch, labels, batch_size):
    if labels[0] in batch.columns:
        labels = batch[labels]
    else:
        labels = np.zeros(shape=(batch.shape[0], len(labels)))
    comments_text = batch['comment_text'].fillna('none')
    comments_ids = to_word_id(comments_text)
    comments = padding(comments_ids, batch_size, SENTENCE_LENGTH)
    data = np.concatenate((comments, labels), axis=1)
    return data

def preprocess_char(batch, labels, batch_size):
    if labels[0] in batch.columns:
        labels = batch[labels]
    else:
        labels = np.zeros(shape=(batch.shape[0], len(labels)))
    comments_text = batch['comment_text'].fillna('none')
    comments_char_ids = [[[ord(char) for char in tok] for tok in tokenize(sentence)] for sentence in batch]
    comments = padding(comments_char_ids, batch_size, SENTENCE_LENGTH)
    data = np.concatenate((comments, labels), axis=1)
    return data

def sort_mat_by_row_non_zero_counts(m):
    row_length = (m != 0).sum(axis=1)
    df = pd.DataFrame(m)
    df['len'] = row_length
    df.sort_values(by='len', inplace=True)
    del df['len']
    return df.values

class Batch:
    def __init__(self, df, labels, batch_size=100):
        self.df = df
        self.index = 0
        self.batch_size = batch_size
        self.max_sentence_length = SENTENCE_LENGTH
        data, labels = self.preprocess(self.df, labels)
        self.data, self.labels = self.sort_mat_by_row_non_zero_counts(data, labels)

    def to_id(self, batch):
        batch = [[dictionary[tok] if tok in dictionary else dictionary[unseen] for tok in tokenize(sentence)] for sentence in batch]
        return batch

    def padding(self, batch):
        batch = np.array(batch)
        zeros = np.zeros((batch.shape[0], self.max_sentence_length))
        for i in range(batch.shape[0]):
            batch[i] = batch[i][:min(self.max_sentence_length, len(batch[i]))]
            zeros[i, :len(batch[i])] = batch[i]
        batch = zeros
        return batch

    def preprocess(self, batch, labels):
        if labels[0] in batch.columns:
            labels = batch[labels]
        else:
            labels = pd.DataFrame(np.zeros(shape=(batch.shape[0], len(labels))))
        comments_text = batch['comment_text'].fillna('none')
        comments_ids = self.to_id(comments_text)
        comments = self.padding(comments_ids)
        return comments, labels

    def sort_mat_by_row_non_zero_counts(self, m, labels):
        axes = tuple(x for x in range(2, len(m.shape)))
        row_length = np.sum(np.sum(m, axis=axes) != 0, axis=1)
        ori_shape = m.shape
        df = pd.DataFrame(np.reshape(m, newshape=(m.shape[0], np.prod(np.array(m.shape[1:])))))
        df['len'] = row_length
        labels_col = ['label_%d'%(x) for x in range(labels.shape[1])]
        labels.columns = labels_col
        #ll = pd.DataFrame(labels, columns=labels_col)
        df = pd.concat([df, labels], axis=1)
        df.sort_values(by='len', inplace=True)
        ll = df[labels_col]
        df.drop(labels_col + ['len'], axis=1, inplace=True)
        m = np.reshape(df.values, newshape=ori_shape)
        return m, ll

    def __next__(self):
        try:
            if self.index >= self.df.shape[0]:
                raise IndexError

            next_index = self.index+self.batch_size
            if next_index > self.df.shape[0]:
                next_index = self.df.shape[0]

            data, label = self.data[self.index:next_index, :SENTENCE_LENGTH], self.labels[self.index:next_index]

            sentence_length = (data.sum(axis=0) != 0).sum()
            data = data[:, :sentence_length]

            self.index = next_index
            return data, label
        except IndexError:
            self.index = 0
            #self.data = np.random.permutation(self.data)
            raise StopIteration

    def __iter__(self):
        return self

class Batch_char(Batch):
    def __init__(self, df, labels, batch_size=100):
        self.max_char_length = CHAR_LENGTH
        Batch.__init__(self, df, labels, batch_size=batch_size)

    def to_id(self, batch):
        batch = [[[ord(char) for char in tok] for tok in tokenize(sentence)] for sentence in batch]
        return batch

    def padding(self, batch):
        zeros = np.zeros((len(batch), self.max_sentence_length, self.max_char_length))
        for i in range(len(batch)):
            sentence_length = min(len(batch[i]), self.max_sentence_length)
            for j in range(sentence_length):
                char_length = min(len(batch[i][j]), self.max_char_length)
                zeros[i, j, :char_length] = batch[i][j][:char_length]
        batch = zeros
        return batch

    def __next__(self):
        try:
            if self.index >= self.df.shape[0]:
                raise IndexError

            next_index = self.index+self.batch_size
            if next_index > self.df.shape[0]:
                next_index = self.df.shape[0]

            #self.batch = self.df.iloc[self.index:next_index, :]
            data, label = self.data[self.index:next_index, :, :], self.labels[self.index:next_index]

            sentence_length = (data != 0).sum(axis=1).max()
            char_length = (data != 0).sum(axis=2).max()
            data = data[:, :sentence_length, :char_length]

            self.index = next_index
            return data, label
        except IndexError:
            self.index = 0
            #self.data = np.random.permutation(self.data)
            raise StopIteration

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

def make_binary_label(df, cat):
    df['label'] = 0
    df.loc[df[cat] == 1, 'label'] = 1
    labels = df['label']
    tmp = np.zeros(shape=(labels.shape[0], 2))
    tmp[np.arange(tmp.shape[0]), labels] = 1
    df['negative'] = tmp[:, 0]
    df['positive'] = tmp[:, 1]

def over_sampling(data):
    data['id'] = data.index
    ids = data['id'].values.reshape((-1, 1))

    ros = RandomOverSampler()
    #ros = SMOTE()
    over_ids, over_label = ros.fit_sample(ids, data['label'])
    over_ids = pd.DataFrame(over_ids.reshape([over_ids.shape[0]]), columns=['id'])
    over_data = over_ids.merge(data, on='id')
    return over_data

if __name__ == '__main__':
    #test_embed()
    test_batch()
