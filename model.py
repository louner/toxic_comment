import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from common import *

model_filepath = './models/model'
embedding_size = 300
epoch_number = 10
kernel_height = 2
vocab_shape = (20002, 300)
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
batch_size = 10
gru_hidden_size = 128
attention_size = 32

class NNModel:
    def __init__(self, data, ops='train'):
        self.feed_dict = {}
        self.ops = ops
        self.data = data
        
    def start(self):
        with tf.Session() as sess:
            self.sess = sess
            graph = self.build_graph()
            if self.ops == 'train':
                train_set, val_set = self.data
                self.train(graph, train_set, val_set)
                
            if self.ops == 'predict':
                test_set = data
                self.predict(graph, self.sess, test_set)

    # cnn as example
    def build_network(self):
        W = tf.get_variable(name='W', shape=vocab_shape, trainable=True)

        fc_w = tf.Variable(tf.random_normal([fc_dim, num_classes]), dtype=tf.float32)
        fc_b = tf.Variable(tf.random_normal([1, num_classes]), dtype=tf.float32)

        self.dropout_prob = tf.placeholder(dtype=tf.float32)

        embedding = tf.nn.embedding_lookup(params=W, ids=self.inputs, name='embed')
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
        dropped = tf.nn.dropout(output, self.dropout_prob)
        
        predict_layer = tf.tanh(tf.matmul(dropped, fc_w) + fc_b)
        predict = tf.argmax(predict_layer, axis=1)

        saver = tf.train.Saver(var_list=[W])
        saver.restore(self.sess, './models/embed_matrix.ckpt')
        
        return [predict_layer, predict]
                
    def build_graph(self):
        self.labels = tf.placeholder(shape=(None, num_classes), dtype=tf.float32)
        self.inputs = tf.placeholder(shape=(None, None), dtype=tf.int32)
        
        network = self.build_network()
        network_output = network[0]
        
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=network_output)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

        return network[1], train_step, loss, network
    
    def train(self, graph, train_set, val_set):
        predict, train_step, loss = graph[:3]

        writer = tf.summary.FileWriter('summary', self.sess.graph)
        saver = tf.train.Saver()

        bt = Batch(train_set, preprocess, batch_size=batch_size)
        val_bt = Batch(val_set, preprocess, batch_size=batch_size)

        metrics = []
        for epoch in range(epoch_number):
            train_losses = []
            self.feed_dict[self.dropout_prob] = dropout_probability
            
            for batch in bt:
                try:
                    comments_batch, label = batch
                    self.feed_dict[self.inputs] = comments_batch
                    self.feed_dict[self.labels] = label
                    
                    _, l, p = self.sess.run([train_step, loss, predict], feed_dict=self.feed_dict)

                    train_losses.append(l.sum())
                    logger.info('training %f, %s %s'%(l.sum()/batch_size, str(np.unique(p, return_counts=True)), str(label.sum())))
                except KeyboardInterrupt:
                    raise

                except:
                    import traceback
                    logger.error(traceback.format_exc())

            self.feed_dict[self.dropout_prob] = 1.0
            val_losses = []
            for batch in val_bt:
                comments_batch, label = batch
                self.feed_dict[self.inputs] = comments_batch
                self.feed_dict[self.labels] = label
                
                l = self.sess.run(loss, feed_dict=self.feed_dict)
                val_losses.append(l.sum())

            saver.save(self.sess, '%s_%d'%(model_filepath, epoch))
            logger.info('%d epoch avg loss: %f'%(epoch, sum(val_losses)))
            metrics.append([sum(train_losses)/len(train_losses), sum(val_losses)/len(val_losses)])

            df_metrics = pd.DataFrame(metrics, columns=['train_err', 'val_err'])
            df_metrics['x'] = df_metrics.index
            df_metrics.to_csv('log/metrics.csv')

if __name__ == '__main__':
    train_set = pd.read_csv('data/train_over_sampled.csv').sample(frac=1)
    val_set = pd.read_csv('data/val_split.csv')
    data = [train_set, val_set]

    cnn = NNModel(data, ops='train')
    cnn.start()
