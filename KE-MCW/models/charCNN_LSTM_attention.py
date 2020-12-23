import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()


class myModel(object):

    def __init__(self,
                 nh1,   # nh1表示第1层rnn神经元的个数
                 nh2,   # nh2表示第2层rnn神经元的个数
                 ny,    # ny: 第1层rnn输出的类别数
                 nz,    # nz: 第2层rnn输出的类别数
                 de,    # word_emb_dimension: 300
                 lr,    # 学习率
                 lr_decay,
                 word_embedding, # 词向量
                 char_embedding,# 字符向量
                 max_gradient_norm,
                 keep_prob,
                 rnn_model_cell='rnn',
                 nonstatic=False):
        self.batch_size = 16
        self.input_word_idx = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='input_word_idx')  # input_word_idx.shape=(None,None)  (16,每段文本单词数)
        self.input_char_idx = tf.compat.v1.placeholder(tf.int32, shape=[None, None,None], name='input_char_idx')  # input_word_idx.shape=(None,None)  (16,每段文本单词数, 每个单词字母数)
        self.rnn_input_y = tf.compat.v1.placeholder(tf.int32, shape=[None, None],  name="rnn_input_y")  # rnn_input_y.shape = (None,None)
        self.rnn_input_z = tf.compat.v1.placeholder(tf.int32, shape=[None, None],  name='rnn_input_z')  # rnn_input_z.shape = (None,None)
        self.keep_prob = keep_prob
        self.lr = tf.Variable(lr, dtype=tf.float32)
        self.learning_rate_decay_op = self.lr.assign(self.lr * lr_decay)
        # Creating wordembedding input
        with tf.device("/cpu:0"), tf.name_scope('embedding'):
            if nonstatic:
                word_emb = tf.constant(word_embedding, name='word_emb', dtype=tf.float32)
                char_emb = tf.constant(char_embedding, name='char_emb', dtype=tf.float32)
            else:
                word_emb = tf.Variable(word_embedding, name='word_emb', dtype=tf.float32)
                char_emb = tf.Variable(char_embedding, name='char_emb', dtype=tf.float32)
        word_emb_inputs = tf.nn.embedding_lookup(word_emb, self.input_word_idx)
        word_emb_inputs = tf.reshape(word_emb_inputs, [self.batch_size, -1, de])  # (16,?,300)
        char_emb_inputs = tf.nn.embedding_lookup(char_emb, self.input_char_idx)
        char_emb_inputs = tf.reshape(char_emb_inputs, [self.batch_size, -1, 20, 30])    #(batch_size, 一段文本的单词数, 每个单词的最大字母数, 每个字母的维度)
            # cnn_inputs = tf.nn.embedding_lookup(word_emb, self.cnn_input_x)
            # cnn_inputs = tf.reshape(cnn_inputs, [self.batch_size, -1, de, 1])
            # word_emb_inputs = tf.nn.embedding_lookup(word_emb, self.cnn_input_x)
            # word_emb_inputs = tf.reshape(word_emb_inputs, [self.batch_size, -1, de])  # (16,?,300)

        self.conv = tf.layers.conv2d(
            inputs=char_emb_inputs,
            filters=30,
            kernel_size=[1, 3],
            strides=[1, 1],
            padding='same',
            activation=tf.nn.relu)
        # word_max_len = self.conv.shape[2]
        charCNN_output = tf.layers.max_pooling2d(inputs=self.conv, pool_size=(1, 20), strides=[1,20])    # (batch_size, 一段文本的单词数, 1, 每个字母的维度)
        charCNN_output = tf.reshape(charCNN_output, [self.batch_size, -1, 30])  #（16, ?, 30）
        rnn_inputs = tf.concat([word_emb_inputs, charCNN_output], 2)    # (16, ?, 330)
        # Droupout embedding input
        rnn_inputs = tf.nn.dropout(rnn_inputs, rate=1 - self.keep_prob, name='rnn_inputs')

        # Create the internal multi-layer cell for rnn
        if rnn_model_cell == 'rnn':
            single_cell1 = tf.nn.rnn_cell.BasicRNNCell(nh1) # nh1表示神经元的个数,330
            single_cell2 = tf.nn.rnn_cell.BasicRNNCell(nh2) # nh2表示神经元的个数,330
        elif rnn_model_cell == 'lstm':
            single_cell1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(nh1, state_is_tuple=True)
            single_cell2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(nh2, state_is_tuple=True)
        elif rnn_model_cell == 'gru':
            single_cell1 = tf.nn.rnn_cell.GRUCell(nh1)
            single_cell2 = tf.nn.rnn_cell.GRUCell(nh2)
        else:
            raise Exception('model_cell error!')
        # DropoutWrapper rnn_cell
        self.single_cell1 = tf.compat.v1.nn.rnn_cell.DropoutWrapper(single_cell1, output_keep_prob=self.keep_prob)
        self.single_cell2 = tf.compat.v1.nn.rnn_cell.DropoutWrapper(single_cell2, output_keep_prob=self.keep_prob)
        self.init_state = self.single_cell1.zero_state(self.batch_size, dtype=tf.float32)

        # RNN1
        with tf.compat.v1.variable_scope('rnn1'):
            # rnn_conv_1
            self.rnn_outputs1, self.rnn_state1 = tf.compat.v1.nn.dynamic_rnn(
                cell=self.single_cell1,
                inputs=rnn_inputs,
                initial_state=self.init_state,
                dtype=tf.float32
            )

        # RNN2
        with tf.compat.v1.variable_scope('rnn2'):
            # rnn_conv_2
            self.rnn_outputs2, self.rnn_state2 = tf.compat.v1.nn.dynamic_rnn(
                cell=self.single_cell2,
                inputs=self.rnn_outputs1,
                initial_state=self.init_state,
                dtype=tf.float32
            )

        # outputs_y
        with tf.compat.v1.variable_scope('output_sy'):
            w_y = tf.compat.v1.get_variable("softmax_w_y", [nh1, ny])  # w_y (400, 2)
            b_y = tf.compat.v1.get_variable("softmax_b_y", [ny])  # b_y (2, )
            rnn_outputs1 = tf.reshape(self.rnn_outputs1, [-1, nh1])  # rnn_ori_outputs1 (?, 400)
            sy = tf.compat.v1.nn.xw_plus_b(rnn_outputs1, w_y, b_y)  # sy (?, 2)
            self.sy_pred = tf.reshape(tf.argmax(sy, 1), [self.batch_size, -1])  # sy_pred (16, ?)
        # outputs_z
        with tf.compat.v1.variable_scope('output_sz'):
            w_z = tf.get_variable("softmax_w_z", [nh2, nz])  # w_z (400, 5)
            b_z = tf.get_variable("softmax_b_z", [nz])  # b_z (5, )
            rnn_outputs2 = tf.reshape(self.rnn_outputs2, [-1, nh2])  # rnn_ori_outputs2 (?, 400)
            sz = tf.compat.v1.nn.xw_plus_b(rnn_outputs2, w_z, b_z)  # sz (?, 5)
            self.sz_pred = tf.reshape(tf.argmax(sz, 1), [self.batch_size, -1])  # sz_pred (16, ?)
        # loss
        with tf.compat.v1.variable_scope('loss'):
            label_y = tf.reshape(self.rnn_input_y, [-1])  # label_y (?, )
            # label_y = tf.argmax(self.rnn_input_y, 1)
            # loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(sy, label_y)
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_y, logits=sy)  # loss1 (?, )
            label_z = tf.reshape(self.rnn_input_z, [-1])  # label_z (?, )
            # label_z = tf.argmax(self.rnn_input_z, 1)
            # loss2 = tf.nn.softmax_cross_entropy_with_logits(labels=sz, logits=label_z)
            loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_z, logits=sz)  # loss2 (?, )
            self.loss = tf.reduce_sum(0.5 * loss1 + 0.5 * loss2) / tf.cast(self.batch_size, tf.float32)

        tvars = tf.compat.v1.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_gradient_norm)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.lr)
        # grads_and_vars = optimizer.compute_gradients(self.loss)     ################################
        # self.train_op=optimizer.apply_gradients(list(zip(grads,tvars)))
        self.train_op = optimizer.minimize(self.loss)

    def cost(output, target):
        # Compute cross entropy for each frame.
        cross_entropy = target * tf.log(output)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
        return tf.reduce_mean(cross_entropy)


