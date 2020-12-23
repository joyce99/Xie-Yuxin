import numpy as np
from attention import attention
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

class Model(object):
    def __init__(self,
                 nh1,   # nh1表示第1层rnn神经元的个数450
                 nh2,   # nh2表示第2层rnn神经元的个数450
                 ny,    # ny: 第1层rnn输出的类别数
                 nz,    # nz: 第2层rnn输出的类别数
                 de,    # emb_dimension: 300
                 cs,
                 lr,    # 学习率
                 lr_decay,
                 embedding, # 词向量
                 max_gradient_norm,
                 batch_size,
                 model_cell='lstm',
                 nonstatic=False):
        self.batch_size = batch_size
        self.input_x = tf.compat.v1.placeholder(tf.int32, shape=[None, None, cs],name='input_x')  # input_x.shape=(None,None,3)
        self.input_y = tf.compat.v1.placeholder(tf.int32, shape=[None, None],name="input_y")  # input_y.shape = (None,None)
        self.input_z = tf.compat.v1.placeholder(tf.int32, shape=[None, None],name='input_z')  # input_z.shape = (None,None)
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)

        self.lr = tf.Variable(lr, dtype=tf.float32)

        self.learning_rate_decay_op = self.lr.assign(self.lr * lr_decay)

        # Creating embedding input
        with tf.device("/cpu:0"), tf.name_scope('embedding'):
            if nonstatic:
                W = tf.constant(embedding, name='embW', dtype=tf.float32)
            else:
                W = tf.Variable(embedding, name='embW', dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(W, self.input_x)
            inputs = tf.reshape(inputs, [self.batch_size, -1, cs * de])  # (16,?,900)

        inputs = tf.nn.dropout(inputs, rate=1 - self.keep_prob, name='drop_inputs')

        with tf.device("/gpu:0"):
            # Create the internal multi-layer cell for rnn
            if model_cell == 'rnn':
                single_cell1 = tf.nn.rnn_cell.BasicRNNCell(nh1) # nh1表示神经元的个数,450
                single_cell2 = tf.nn.rnn_cell.BasicRNNCell(nh2) # nh2表示神经元的个数,450
            elif model_cell == 'lstm':
                single_cell1 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(nh1, state_is_tuple=True)
                single_cell2 = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(nh2, state_is_tuple=True)
            elif model_cell == 'gru':
                single_cell1 = tf.nn.rnn_cell.GRUCell(nh1)
                single_cell2 = tf.nn.rnn_cell.GRUCell(nh2)
            else:
                raise Exception('model_cell error!')
            # DropoutWrapper rnn_cell
            self.single_cell1 = tf.compat.v1.nn.rnn_cell.DropoutWrapper(single_cell1, output_keep_prob=self.keep_prob)
            self.single_cell2 = single_cell2
            # self.single_cell2 = tf.compat.v1.nn.rnn_cell.DropoutWrapper(single_cell2, output_keep_prob=self.keep_prob)
            self.init_state = self.single_cell1.zero_state(self.batch_size, dtype=tf.float32)

            # RNN1
            with tf.compat.v1.variable_scope('rnn1'):
                # rnn_conv_1
                self.rnn_outputs1, self.rnn_state1 = tf.compat.v1.nn.dynamic_rnn(
                # self.rnn_conv_outputs1, self.rnn_conv_state1 = tf.compat.v1.nn.dynamic_rnn(     # self.rnn_conv_outputs1 (16, ?, 450), self.rnn_conv_state1 (16, 450)
                    cell=self.single_cell1,
                    inputs=inputs,
                    initial_state=self.init_state,
                    dtype=tf.float32
                )
            # Attention layer1
            # attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs,
            #                                                            memory_sequence_length=encoder_inputs_length)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=nh1, memory=self.rnn_outputs1)

            att_cell = tf.contrib.seq2seq.AttentionWrapper(cell=self.single_cell2, attention_mechanism=attention_mechanism,
                                                               attention_layer_size=nh2, name='Attention_Wrapper')
            # with tf.name_scope('Attention_layer'):
            #     attention_output1, alphas1 = attention(self.rnn_conv_outputs1, ATTENTION_SIZE, return_alphas=True)
            #     tf.summary.histogram('alphas', alphas1)
            #
            # Dropout Attention1
            self.att1_out = tf.compat.v1.nn.rnn_cell.DropoutWrapper(att_cell, output_keep_prob=self.keep_prob)
            self.att_initial_state = self.att1_out.zero_state(batch_size=self.batch_size, dtype=tf.float32)

            # RNN2
            with tf.compat.v1.variable_scope('rnn2'):
                # rnn_conv_2
                self.rnn_out2, self.rnn_state2 = tf.compat.v1.nn.dynamic_rnn(
                    cell=self.att1_out,
                    inputs=self.rnn_outputs1,
                    initial_state=self.att_initial_state,
                    dtype=tf.float32
                )
                # rnn_conv_2 old
                # self.rnn_conv_outputs2, self.rnn_conv_state2 = tf.compat.v1.nn.dynamic_rnn(
                #     cell=self.single_cell2,
                #     inputs=self.rnn_conv_outputs1,
                #     initial_state=self.init_state,
                #     dtype=tf.float32
                # )

            # Attention layer2
            # with tf.name_scope('Attention_layer'):
            #    attention_output2, alphas2 = attention(self.rnn_conv_outputs2, ATTENTION_SIZE, return_alphas=True)
            #    tf.summary.histogram('alphas', alphas2)
            #
            # Dropout
            # self.att2_out = tf.nn.dropout(attention_output2, rate=1-0.8, name='drop_att_2')
            # self.rnn_out2 = tf.nn.dropout(self.rnn_out2, rate=1 - 0.8, name='drop_att_2')

            # outputs_y
            with tf.compat.v1.variable_scope('output_sy'):
                w_y = tf.compat.v1.get_variable("softmax_w_y", [nh1, ny])  # w_y (450, 2)
                b_y = tf.compat.v1.get_variable("softmax_b_y", [ny])  # b_y (2, )
                rnn_outputs1 = tf.reshape(self.rnn_outputs1, [-1, nh1])  # rnn_ori_outputs1 (?, 450)
                sy = tf.compat.v1.nn.xw_plus_b(rnn_outputs1, w_y, b_y)  # sy (?, 2)
                self.sy_pred = tf.reshape(tf.argmax(sy, 1), [self.batch_size, -1])  # sy_pred (16, ?)
            # outputs_z
            with tf.compat.v1.variable_scope('output_sz'):
                w_z = tf.get_variable("softmax_w_z", [nh2, nz])  # w_z (450, 5)
                b_z = tf.get_variable("softmax_b_z", [nz])  # b_z (5, )
                rnn_outputs2 = tf.reshape(self.rnn_out2, [-1, nh2])  # rnn_ori_outputs2 (?, 450) ######################################
                sz = tf.compat.v1.nn.xw_plus_b(rnn_outputs2, w_z, b_z)  # sz (?, 5)
                self.sz_pred = tf.reshape(tf.argmax(sz, 1), [self.batch_size, -1])  # sz_pred (16, ?)
            # loss
            with tf.compat.v1.variable_scope('loss'):
                label_y = tf.reshape(self.input_y, [-1])  # label_y (?, )
                loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_y, logits=sy)  # loss1 (?, )
                label_z = tf.reshape(self.input_z, [-1])  # label_z (?, )
                loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_z, logits=sz)  # loss2 (?, )
                self.loss = tf.reduce_sum(0.5 * loss1 + 0.5 * loss2) / tf.cast(self.batch_size, tf.float32)

            tvars = tf.compat.v1.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_gradient_norm)
            #optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.lr)
            optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
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


