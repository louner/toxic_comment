import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from common import *
from ggplot import *

from time import time
#np.random.seed(0)

embedding_size = 300
kernel_height = 2
vocab_shape = (20001, 300)
kernel_number = 32
kernel_size = (kernel_height, embedding_size)
stack_kernel_size = (kernel_height, kernel_number)
num_classes = len(categories)
learning_rate = 0.001
epoch_number = 200
layner_number = 2
epslon = 1e-10
fc_dim = SENTENCE_LENGTH*kernel_number
kernel_heights = [x for x in range(2, 5)]
fc_dim = kernel_number*len(kernel_heights)
dropout_probability = 0.2

gru_hidden_size = 128
attention_size = 32

def build_network_test(x, W):
    # load embedding W
    embedding = tf.nn.embedding_lookup(params=W, ids=x, name='embed')

    # reshape to NHWC
    # H -> longest sentence length
    reshape1 = tf.reshape(embedding, shape=(-1, SENTENCE_LENGTH, embedding_size, 1), name='reshape_first')

    # pad at both ends of a sentence
    pad1 = tf.pad(reshape1, [[0, 0], [kernel_height - 1, kernel_height - 1], [0, 0], [0, 0]], name='padding_first')

    # output: batch_size, longest_length+(kernel_height-1)*2-(kernel_hefight-1), 1, kernel_number
    # w-op
    conv1 = tf.layers.conv2d(pad1,
                            filters=kernel_number,
                            kernel_size=kernel_size,
                            strides=(1, 1),
                            padding='valid',
                            activation=tf.tanh,
                            name='conv_first')

    # output: batch_size, longest_length, 1, kernel_number
    pool1 = tf.nn.pool(conv1,
                      window_shape=(kernel_height, 1),
                      pooling_type='AVG',
                      padding='VALID',
                    name='pool_first')

    output = tf.reshape(pool1, shape=(-1, fc_dim))
    return output, embedding

def build_network_cnn(x, W, dropout_prob):
    embedding = tf.nn.embedding_lookup(params=W, ids=x, name='embed')
    reshape1 = tf.reshape(embedding, shape=(-1, SENTENCE_LENGTH, embedding_size, 1), name='reshape_first')

    conv_layer, pool_layer = [], []
    for kernel_height in kernel_heights:
        with tf.variable_scope('conv_%d'%(kernel_height)):
            conv = tf.layers.conv2d(reshape1,
                                    filters=kernel_number,
                                    kernel_size=(kernel_height, embedding_size),
                                    strides=(1, 1),
                                    padding='valid',
                                    activation=tf.tanh,
                                    name='conv_%d'%(kernel_height))

            pool = tf.nn.max_pool(conv,
                                  (1, SENTENCE_LENGTH-kernel_height+1, 1, 1),
                                  (1, 1, 1, 1),
                                  padding='VALID',
                                  data_format='NHWC',
                                  name='pool_%d'%(kernel_height))
        pool_layer.append(pool)

    concat = tf.concat(pool_layer, axis=1)
    output = tf.reshape(concat, (-1, fc_dim))
    dropped = tf.nn.dropout(output, dropout_prob)
    return dropped

def build_network_gru(x, W, dropout_prob):
    embedding = tf.nn.embedding_lookup(params=W, ids=x, name='embed')
    reshape = tf.reshape(embedding, shape=(-1, SENTENCE_LENGTH, embedding_size), name='reshape')

    # RNN input form: [sentence_length, [batch_size, embedding_size]]
    inputs = tf.unstack(value=reshape,
                        num=SENTENCE_LENGTH,
                        axis=1)

    gru = rnn.GRUCell(num_units=gru_hidden_size)
    outputs, last_h = rnn.static_rnn(cell=gru,
                                     inputs=inputs,
                                     dtype=tf.float32)
    return outputs, last_h

def build_network(x, W, dropout_prob):
    ws = tf.Variable(tf.random_normal((gru_hidden_size, attention_size)), name='w_s')
    bs = tf.Variable(tf.random_normal((1, 1, attention_size), name='b_s'))

    us = tf.Variable(tf.random_normal((attention_size, 1)), name='u_s')

    outputs, last_h = build_network_gru(x, W, dropout_prob)

    outputs = tf.einsum('ijk->jik', tf.stack(outputs))

    ui = tf.tanh(tf.tensordot(outputs, ws, axes=[[2], [0]]))

    context = tf.exp(tf.tensordot(ui, us, axes=[[2], [0]]))
    context = tf.reshape(context, shape=(-1, SENTENCE_LENGTH))
    attention_weight = tf.nn.softmax(context, dim=1)

    attension_weighted_output = tf.einsum('ijk,ij->ik', outputs, attention_weight)

    return attension_weighted_output

def build_graph(sess):
    W = tf.get_variable(name='W', shape=vocab_shape, trainable=True)

    fc_w = tf.Variable(tf.random_normal([gru_hidden_size, num_classes]), dtype=tf.float32)
    fc_b = tf.Variable(tf.random_normal([1, num_classes]), dtype=tf.float32)

    # batch size and sentence length is unknown
    # tensor for longest sentence length of each batch
    labels = tf.placeholder(shape=(None, num_classes), dtype=tf.float32)
    comments = tf.placeholder(shape=(None, None), dtype=tf.int32)
    dropout_prob = tf.placeholder(dtype=tf.float32)

    # build network
    #network_output = build_network(comments, W, dropout_prob)
    _, network_output = build_network_gru(comments, W, dropout_prob)

    predict_layer = tf.matmul(network_output, fc_w) + fc_b

    predict_prob = tf.nn.softmax(predict_layer)

    #loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predict_layer)
    loss = tf.reduce_sum(tf.log(tf.nn.softmax(predict_layer + epslon)) * labels) * -1
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    predict = tf.argmax(predict_layer, axis=1)

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=[W])
    saver.restore(sess, './models/embed_matrix.ckpt')

    return predict, predict_prob, loss, train_step, comments, labels, dropout_prob, network_output

def train(train_set, val_set, model_filepath, epoch_number, graphs, sess, do_self_evaluation=True):
    predict, predict_prob, loss, train_step, comments, labels, dropout_prob, network_output = graphs

    writer = tf.summary.FileWriter('summary', sess.graph)
    saver = tf.train.Saver()

    bt = Batch(train_set, preprocess, batch_size=batch_size)
    val_bt = Batch(val_set, preprocess, batch_size=batch_size)

    metrics = []
    for epoch in range(epoch_number):
        st = time()
        train_losses = []
        for batch in bt:
            try:
                comments_batch, label = batch
                _, l, p = sess.run([train_step, loss, predict], feed_dict={comments: comments_batch, labels: label, dropout_prob: dropout_probability})
                #e = sess.run(embedding, feed_dict={comments: comments_batch, labels: label})

                train_losses.append(l.sum())
                logger.info('training %f, %s %s'%(l.sum()/batch_size, str(np.unique(p, return_counts=True)), str(label.sum())))
            except KeyboardInterrupt:
                raise

            except:
                import traceback
                logger.error(traceback.format_exc())

        val_losses = []
        for batch in val_bt:
            comments_batch, label = batch
            l = sess.run(loss, feed_dict={comments: comments_batch, labels: label, dropout_prob: 1.0})
            val_losses.append(l.sum())

        saver.save(sess, '%s_%d'%(model_filepath, epoch))
        logger.info('%d epoch avg loss: %f'%(epoch, sum(val_losses)))
        metrics.append([sum(train_losses)/len(train_losses), sum(val_losses)/len(val_losses)])

        df_metrics = pd.DataFrame(metrics, columns=['train_err', 'val_err'])
        df_metrics['x'] = df_metrics.index
        df_metrics.to_csv('log/metrics.csv')


if __name__ == '__main__':
    train_set = pd.read_csv('data/train_over_sampled.csv')
    val_set = pd.read_csv('data/val_split.csv')
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
        graphs = build_graph(sess)
        train(train_set, val_set, './models/model', epoch_number, graphs, sess)