import tensorflow as tf
from tensorflow.contrib import rnn, layers
import numpy as np
from common import *
from sklearn.model_selection import KFold
from time import sleep, time

model_filepath = './models/model'
embedding_size = 50
epoch_number = 10
kernel_height = 2
vocab_shape = (72587, embedding_size)
kernel_number = 64
kernel_size = (kernel_height, embedding_size)
stack_kernel_size = (kernel_height, kernel_number)
num_classes = len(categories)
#num_classes = 2
learning_rate = 0.01
epoch_number = 50
layner_number = 2
epslon = 1e-10
fc_dim = SENTENCE_LENGTH*kernel_number
kernel_heights = [x for x in range(2, 5)]
fc_dim = kernel_number*len(kernel_heights)
dropout_keep_probability = 1.0
batch_size = 128
gru_hidden_size = 32

attention_size = 32

rnn_layer_num = 4

char_embedding_size = 32

time_stamp = str(time())

class NNModel:
    def __init__(self, model_name, labels, **kwargs):
        self.summary_path = 'base'
        self.feed_dict = {}
        self.model_name = '%s_%s'%(self.__class__.__name__, model_name)
        self.model_filepath = 'models/%s'%(self.model_name)
        self.metric_path = 'metrics/%s'%(self.model_name)
        self.submit_path = 'submit/%s'%(self.model_name)
        self.logger = init_logger('log/%s'%(self.model_name))
        self.Labels = labels

        self.vocab_shape = vocab_shape
        self.fc_dim = fc_dim
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.epoch_number = epoch_number
        self.dropout_keep_probability = dropout_keep_probability

        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(self, data):
        tf.reset_default_graph()
        self.feed_dict = {}
        with tf.Session() as sess:
            self.sess = sess
            self.build_graph()

            train_set, val_set = data
            self._train(train_set, val_set, self.Labels)

    def predict(self, data):
        self.feed_dict = {}
        tf.reset_default_graph()
        with tf.Session() as sess:
            self.sess = sess
            graph = self.build_graph()

            test_set = data
            self._predict(graph, test_set)

    # cnn as example
    def build_network(self):
        W = tf.get_variable(name='W', shape=self.vocab_shape, trainable=True)

        fc_w = tf.Variable(tf.random_normal([self.fc_dim, len(self.Labels)]), dtype=tf.float32)
        fc_b = tf.Variable(tf.random_normal([1, len(self.Labels)]), dtype=tf.float32)

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
        dropped = tf.nn.dropout(output, self.dropout_keep_prob)
        
        self.predict_layer = tf.nn.sigmoid(tf.matmul(dropped, fc_w) + fc_b)
        self.predict_prob = tf.nn.softmax(self.predict_layer)
        self.predict = tf.argmax(self.predict_layer, axis=1)

        saver = tf.train.Saver(var_list=[W])
        saver.restore(self.sess, './models/embed_matrix.ckpt')

    def build_loss(self):
        #self.loss = tf.norm(self.labels-self.predict_layer, ord=2)
        #return -1 * tf.reduce_sum(tf.multiply(labels, tf.log(tf.clip_by_value(logits, 1e-3, 1e3))))
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.predict_layer)
        #self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.predict)

    def build_inputs(self):
        self.labels = tf.placeholder(shape=(None, len(self.Labels)), dtype=tf.float32, name='label')
        self.inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inputs')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32)
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.sentence_length = tf.placeholder(dtype=tf.int32)
                
    def build_graph(self):
        self.global_step = tf.Variable(0, name='global_step')

        self.build_inputs()

        self.build_network()

        #self.labels = tf.nn.softmax(self.labels)
        self.build_loss()
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels, axis=1), self.predict), dtype=tf.float32))

        glob_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()
        self.sess.run([glob_init, local_init])

    def build_feed_dict(self, comments_batch, labels):
        self.feed_dict[self.inputs] = comments_batch
        self.feed_dict[self.labels] = labels
        self.feed_dict[self.batch_size] = comments_batch.shape[0]
        self.feed_dict[self.sentence_length] = (comments_batch != 0).sum(axis=1)

    def _train(self, train_set, val_set, labels):
        #check = tf.add_check_numerics_ops()

        writer = tf.summary.FileWriter('summary/%s/%s'%(self.summary_path, time_stamp), self.sess.graph)
        tf.summary.scalar('loss', tf.reduce_mean(self.loss))
        tf.summary.scalar('accuracy', self.acc)
        merged = tf.summary.merge_all()

        saver = tf.train.Saver(max_to_keep=0)

        bt = Batch(train_set, labels, batch_size=batch_size)
        val_bt = Batch(val_set, labels, batch_size=batch_size)

        metrics = []
        train_step, val_step = 0, 0
        for epoch in range(epoch_number):
            self.feed_dict[self.dropout_keep_prob] = dropout_keep_probability

            for batch in bt:
                try:
                    #sleep(1)
                    comments_batch, label = batch
                    seq_len = (comments_batch != 0).sum(axis=1)
                    if comments_batch.shape[0] != batch_size:
                        continue
                    print(comments_batch.shape)

                    self.build_feed_dict(comments_batch, label)

                    _, l, acc = self.sess.run([self.train_step, self.loss, self.acc], feed_dict=self.feed_dict)
                    train_step += 1

                    if train_step % 5 == 0:
                        summary = tf.Summary()
                        summary.value.add(tag='train/loss', simple_value=l.mean())
                        summary.value.add(tag='train/accuracy', simple_value=acc)
                        writer.add_summary(summary, global_step=train_step)
                        writer.flush()

                    #l, p = self.sess.run([loss, predict], feed_dict=self.feed_dict)
                    #self.logger.info('%d: training %f %s %s'%(step, l.sum()/batch_size, pred.sum(), label.sum()))
                except KeyboardInterrupt:
                    raise

                except:
                    import traceback
                    self.logger.error(traceback.format_exc())
                    raise

            self.feed_dict[self.dropout_keep_prob] = 1.0
            val_acc = []
            for batch in val_bt:
                comments_batch, labels = batch
                if comments_batch.shape[0] != batch_size:
                    continue

                self.build_feed_dict(comments_batch, labels)
                print(comments_batch.shape)

                acc = self.sess.run(self.acc, feed_dict=self.feed_dict)
                val_acc.append(acc)

            saver.save(self.sess, '%s_%d'%(self.model_filepath, epoch))

            summary = tf.Summary()
            summary.value.add(tag='val/accuracy', simple_value=mean(val_acc))
            writer.add_summary(summary, global_step=epoch)
            writer.flush()

    def _predict(self, graph, test_set):
        predict_prob = graph[-1][-1]
        bt = Batch(test_set, preprocess, ['none'], batch_size)

        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_filepath)
        self.feed_dict[self.dropout_keep_prob] = 1.0

        predict_probs = []
        for batch in bt:
            comments_batch, label = batch
            self.feed_dict[self.inputs] = comments_batch
            self.feed_dict[self.sentence_length] = comments_batch.shape[1]

            p = self.sess.run(predict_prob, feed_dict=self.feed_dict)
            predict_probs += p.tolist()

        output = pd.DataFrame(predict_probs, columns=self.Labels)
        output['id'] = test_set['id']
        output.to_csv(self.submit_path, index=False)

class GRU(NNModel):
    def __init__(self, model_name, labels, **kwargs):
        NNModel.__init__(self, model_name, labels, **kwargs)
        self.gru_hidden_size = gru_hidden_size
        self.summary_path = 'gru'

    def build_network(self):
        W = tf.get_variable(name='W', shape=vocab_shape)
        #embed_mat = np.random.rand(vocab_shape[0], vocab_shape[1])
        #W = tf.constant(W, dtype=tf.float32)

        with tf.variable_scope(name_or_scope='gru'):
            fc_w_1 = tf.Variable(tf.random_normal([gru_hidden_size, gru_hidden_size]), dtype=tf.float32)
            fc_b_1 = tf.Variable(tf.random_normal([1, gru_hidden_size]), dtype=tf.float32)

            fc_w_2 = tf.Variable(tf.random_normal([gru_hidden_size, len(self.Labels)]), dtype=tf.float32)
            fc_b_2 = tf.Variable(tf.random_normal([1, len(self.Labels)]), dtype=tf.float32)

            self.embedding = tf.nn.embedding_lookup(params=W, ids=self.inputs, name='embed')
            #reshape = tf.reshape(embedding, shape=(-1, self.sentence_length, embedding_size), name='reshape_first')
            reshape = tf.reshape(self.embedding, shape=(-1, SENTENCE_LENGTH, embedding_size), name='reshape_first')

            # RNN input form: [sentence_length, [batch_size, embedding_size]]
            #inputs = tf.unstack(value=reshape, axis=1)
            inputs = reshape

            gru = rnn.GRUCell(num_units=gru_hidden_size)
            outputs, state = tf.nn.dynamic_rnn(gru,
                                               inputs,
                                               sequence_length=self.sentence_length,
                                               dtype=tf.float32)

            index = tf.range(0, batch_size) * SENTENCE_LENGTH + (self.sentence_length - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, gru_hidden_size]), index)  # batch_size * n_hidden
            '''
            outputs, last_h = rnn.static_rnn(cell=gru,
                                                inputs=inputs,
                                                dtype=tf.float32)
            outputs = outputs[-1]
            '''

            dense_layer = tf.nn.relu(tf.matmul(outputs, fc_w_1) + fc_b_1)
            self.predict_layer = tf.matmul(dense_layer, fc_w_2) + fc_b_2
            self.predict = tf.argmax(self.predict_layer, axis=1)
            self.predict_prob = tf.nn.softmax(self.predict_layer, name='predict_prob')

class BILSTM(GRU):
    def __init__(self, model_name, labels, **kwargs):
        GRU.__init__(self, model_name, labels, **kwargs)
        self.summary_path = 'BILSTM'

    def build_network(self):
        W = tf.get_variable(name='W', shape=vocab_shape)

        with tf.variable_scope(name_or_scope='lstm'):
            fc_w_1 = tf.Variable(tf.random_normal([gru_hidden_size*2, gru_hidden_size*2]), dtype=tf.float32)
            fc_b_1 = tf.Variable(tf.random_normal([1, gru_hidden_size*2]), dtype=tf.float32)

            fc_w_2 = tf.Variable(tf.random_normal([gru_hidden_size*2, len(self.Labels)]), dtype=tf.float32)
            fc_b_2 = tf.Variable(tf.random_normal([1, len(self.Labels)]), dtype=tf.float32)

            embedding = tf.nn.embedding_lookup(params=W, ids=self.inputs, name='embed')
            reshape = tf.reshape(embedding, shape=(-1, self.sentence_length, embedding_size), name='reshape_first')
            inputs = reshape

            forward_lstm = rnn.BasicLSTMCell(gru_hidden_size, state_is_tuple=True)
            backword_lstm = rnn.BasicLSTMCell(gru_hidden_size, state_is_tuple=True)

            outputs, (_, forward_last), (_, backword_last) = tf.nn.bidirectional_dynamic_rnn(forward_lstm,
                                                                                       backword_lstm,
                                                                                       inputs,
                                                                                       dtype=tf.float32)

            outputs = tf.concat([forward_last, backword_last], axis=1)
            dense_layer = tf.nn.relu(tf.matmul(outputs, fc_w_1) + fc_b_1)
            self.predict_layer = tf.nn.sigmoid(tf.matmul(dense_layer, fc_w_2) + fc_b_2)
            self.predict = tf.argmax(self.predict_layer, axis=1)
            self.predict_prob = tf.nn.softmax(self.predict_layer)
            #self.predict_prob = tf.Print(self.predict_prob, [tf.shape(forward_last)])

class AttentionGRU(GRU):
    def __init__(self, model_name, labels, **kwargs):
        self.attention_size = attention_size

        GRU.__init__(self, model_name, labels, **kwargs)

    def build_network(self):
        network = GRU.build_network(self)

        ws = tf.Variable(tf.random_normal((gru_hidden_size, attention_size)), name='w_s')
        bs = tf.Variable(tf.random_normal((1, attention_size), name='b_s'))
        us = tf.Variable(tf.random_normal((attention_size, 1)), name='u_s')

        fc_w = tf.Variable(tf.random_normal([gru_hidden_size, len(self.Labels)]), dtype=tf.float32)
        fc_b = tf.Variable(tf.random_normal([1, len(self.Labels)]), dtype=tf.float32)

        outputs, last_h = network[2], network[3]

        _outputs = tf.reshape(outputs, (-1, gru_hidden_size))
        ui = tf.tanh(tf.matmul(_outputs, ws) + bs)
        #ui = tf.tanh(tf.tensordot(outputs, ws, axes=[[2], [0]]))

        context = tf.exp(tf.matmul(ui, us))

        #context = tf.exp(tf.tensordot(ui, us, axes=[[2], [0]]))
        context = tf.reshape(context, shape=(-1, self.sentence_length))
        attention_weight = tf.nn.softmax(context, dim=1)

        attension_weighted_output = tf.reduce_sum(outputs * tf.expand_dims(attention_weight, -1), axis=1)
        #attension_weighted_output = tf.einsum('ijk,ij->ik', outputs, attention_weight)

        predict_layer = tf.nn.tanh(tf.matmul(attension_weighted_output, fc_w) + fc_b)
        predict = tf.argmax(predict_layer, axis=1)
        predict_prob = tf.nn.softmax(predict_layer)

        return [predict_layer, predict, predict_prob]

class Sigmoid(AttentionGRU):
    def build_loss(self, labels, logits):
        return tf.norm(labels-logits, ord=2)

    def build_network(self):
        network = AttentionGRU.build_network(self)
        predict_layer = network[0]

        sigmoid_layer = tf.nn.sigmoid(predict_layer)
        network[0] = sigmoid_layer
        return network

class MultiLayerGRU(GRU):
    def __init__(self, model_name, labels, **kwargs):
        self.attention_size = attention_size

        GRU.__init__(self, model_name, labels, **kwargs)

    def build_network(self):
        W = tf.get_variable(name='W', shape=vocab_shape)

        fc_w_1 = tf.Variable(tf.random_normal([gru_hidden_size, gru_hidden_size]), dtype=tf.float32)
        fc_b_1 = tf.Variable(tf.random_normal([1, gru_hidden_size]), dtype=tf.float32)

        fc_w_2 = tf.Variable(tf.random_normal([gru_hidden_size, len(self.Labels)]), dtype=tf.float32)
        fc_b_2 = tf.Variable(tf.random_normal([1, len(self.Labels)]), dtype=tf.float32)

        embedding = tf.nn.embedding_lookup(params=W, ids=self.inputs, name='embed')
        reshape = tf.reshape(embedding, shape=(-1, SENTENCE_LENGTH, embedding_size), name='reshape_first')
        inputs = reshape

        gru_layers = []
        for _ in range(rnn_layer_num):
            cell = rnn.GRUCell(num_units=gru_hidden_size)
            cell = rnn.DropoutWrapper(cell,
                                      input_keep_prob=self.dropout_keep_prob)

            gru_layers.append(cell)

        cell = rnn.MultiRNNCell(gru_layers)
        outputs, last_h = tf.nn.dynamic_rnn(cell=cell,
                                            inputs=inputs,
                                            sequence_length=self.sentence_length,
                                            dtype=tf.float32)

        index = tf.range(0, batch_size) * SENTENCE_LENGTH + (self.sentence_length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, gru_hidden_size]), index)  # batch_size * n_hidden

        dense_layer = tf.nn.relu(tf.matmul(outputs, fc_w_1) + fc_b_1)
        self.predict_layer = tf.matmul(dense_layer, fc_w_2) + fc_b_2
        self.predict = tf.argmax(self.predict_layer, axis=1)
        self.predict_prob = tf.nn.softmax(self.predict_layer)

class Char_CNN_RNN(NNModel):
    def build_inputs(self):
        NNModel.build_inputs(self)
        self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='inputs')
        self.char_length = tf.placeholder(dtype=tf.int32)

    def build_feed_dict(self, comments_batch, labels):
        NNModel.build_feed_dict(self, comments_batch, labels)
        self.feed_dict[self.char_length] = comments_batch.shape[2]

    def build_network(self):
        w = tf.random_normal(shape=(256, char_embedding_size))
        embed = tf.nn.embedding_lookup(params=w, ids=self.inputs)
        embed = tf.reshape(embed, shape=(-1, self.char_length, char_embedding_size, 1))

        conv_layer, pool_layer = [], []
        for kernel_height in kernel_heights:
            with tf.variable_scope('conv_%d' % (kernel_height)):
                conv = tf.layers.conv2d(embed,
                                        filters=kernel_number,
                                        kernel_size=(kernel_height, char_embedding_size),
                                        strides=(1, 1),
                                        padding='valid',
                                        activation=tf.nn.relu,
                                        name='conv_%d' % (kernel_height))

                pool = tf.nn.max_pool(conv,
                                      (1, CHAR_LENGTH - kernel_height + 1, 1, 1),
                                      (1, 1, 1, 1),
                                      padding='VALID',
                                      data_format='NHWC',
                                      name='pool_%d' % (kernel_height))
            pool_layer.append(pool)

        with tf.variable_scope(name_or_scope='gru', regularizer=layers.l2_regularizer(0.5)):
            concat = tf.concat(pool_layer, axis=1)
            rnn_input = tf.reshape(concat, shape=(self.batch_size, self.sentence_length, kernel_number*len(kernel_heights)))

            fc_w = tf.Variable(tf.random_normal([gru_hidden_size, len(self.Labels)]), dtype=tf.float32)
            fc_b = tf.Variable(tf.random_normal([1, len(self.Labels)]), dtype=tf.float32)

            gru = rnn.GRUCell(num_units=gru_hidden_size)
            outputs, last_h = tf.nn.dynamic_rnn(cell=gru,
                                                inputs=rnn_input,
                                                dtype=tf.float32)

        predict_layer = tf.nn.tanh(tf.matmul(last_h, fc_w) + fc_b)
        predict = tf.argmax(predict_layer, axis=1)
        predict_prob = tf.nn.softmax(predict_layer)

        return [predict_layer, predict, outputs, last_h, predict_prob]

if __name__ == '__main__':
    #df = pd.read_csv('train.csv').sample(frac=1).iloc[:1000, :]
    df = pd.read_csv('train.csv').sample(frac=1.0)
    #df = pd.read_csv('data/train.csv').iloc[:1000, :]
    #df = pd.read_csv('data/train.csv')

    kf = KFold(n_splits=5)
    X = df.values
    kf_times = 0

    for train_index, val_index in kf.split(X):
        train_set, val_set = X[train_index], X[val_index]
        train_set, val_set = pd.DataFrame(train_set, columns=df.columns), pd.DataFrame(val_set, columns=df.columns)

        #make_binary_label(train_set, cat)
        #make_binary_label(val_set, cat)
        #train_set = over_sampling(train_set)

        #model = NNModel('%d'%(kf_times), ['negative', 'positive'])
        model = GRU('%d'%(kf_times), categories)
        #model = NNModel('cnn_toxic', ['negative', 'positive'])
        #model = AttentionGRU(str(kf_times), categories)
        #model = MultiLayerGRU('%d'%(kf_times), categories)
        #model = Sigmoid('sigmoid_%d'%(kf_times), categories)
        #model = Char_CNN_RNN('%d'%(kf_times), categories)
        #model = BILSTM('%d'%(kf_times), categories)
        model.train([train_set, val_set])
        kf_times += 1
        break
