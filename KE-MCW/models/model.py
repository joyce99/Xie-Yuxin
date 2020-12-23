import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
class Model(object):

    def __init__(self,
                 nh1,
                 nh2,
                 ny,
                 nz,
                 de,
                 cs,
                 lr,
                 lr_decay,
                 embedding,
                 max_gradient_norm,
                 batch_size,
                 #keep_prob,
                 model_cell='rnn',
                 #model='basic_model',
                 nonstatic=False):

        # self.batch_size = tf.compat.v1.placeholder(dtype=tf.float32, shape=None)
        # self.input_x=tf.compat.v1.placeholder(tf.int32,shape=[None,None,cs],name='input_x')
        # self.input_y=tf.compat.v1.placeholder(tf.int32,shape=[None,None],name="input_y")
        # self.input_z=tf.compat.v1.placeholder(tf.int32,shape=[None,None],name='input_z')
        # self.keep_prob=tf.compat.v1.placeholder(dtype=tf.float32,name='keep_prob')
        # self.batch_size = 16
        self.batch_size = batch_size
        self.input_x = tf.compat.v1.placeholder(tf.int32, shape=[None, None, cs], name='input_x')   # input_x.shape=(None,None,3)
        self.input_y = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name="input_y") # input_y.shape = (None,None)
        self.input_z = tf.compat.v1.placeholder(tf.int32, shape=[None, None], name='input_z') # input_z.shape = (None,None)
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)

        self.lr=tf.Variable(lr,dtype=tf.float32)

        self.learning_rate_decay_op = self.lr.assign(
            self.lr * lr_decay)


        #Creating embedding input
        with tf.device("/cpu:0"),tf.name_scope('embedding'):
            if nonstatic:
                W=tf.constant(embedding,name='embW',dtype=tf.float32)
            else:
                W=tf.Variable(embedding,name='embW',dtype=tf.float32)
            inputs=tf.nn.embedding_lookup(W,self.input_x)
            inputs=tf.reshape(inputs,[self.batch_size,-1,cs*de])    # (16,?,900)

        #Droupout embedding input
        # inputs=tf.nn.dropout(inputs,keep_prob=self.keep_prob,name='drop_inputs')
        inputs = tf.nn.dropout(inputs, rate=1-self.keep_prob, name='drop_inputs')

        #Create the internal multi-layer cell for rnn
        with tf.device("/gpu:0"):
            if model_cell=='rnn':
                single_cell1=tf.nn.rnn_cell.BasicRNNCell(nh1)
                single_cell2=tf.nn.rnn_cell.BasicRNNCell(nh2)
            elif model_cell=='lstm':
                single_cell1=tf.compat.v1.nn.rnn_cell.BasicLSTMCell(nh1,state_is_tuple=True)
                single_cell2=tf.compat.v1.nn.rnn_cell.BasicLSTMCell(nh2,state_is_tuple=True)
            elif model_cell=='gru':
                single_cell1=tf.nn.rnn_cell.GRUCell(nh1)
                single_cell2=tf.nn.rnn_cell.GRUCell(nh2)
            else:
                raise Exception('model_cell error!')
            #DropoutWrapper rnn_cell
            single_cell1 = tf.compat.v1.nn.rnn_cell.DropoutWrapper(single_cell1, output_keep_prob=self.keep_prob)
            single_cell2 = tf.compat.v1.nn.rnn_cell.DropoutWrapper(single_cell2, output_keep_prob=self.keep_prob)

            self.init_state=single_cell1.zero_state(self.batch_size,dtype=tf.float32)

            #RNN1
            with tf.compat.v1.variable_scope('rnn1'):
                self.outputs1,self.state1=tf.compat.v1.nn.dynamic_rnn(
                    cell=single_cell1,
                    inputs=inputs,
                    initial_state=self.init_state,
                    dtype=tf.float32
                )

            #RNN2
            with tf.compat.v1.variable_scope('rnn2'):
                self.outputs2,self.state2=tf.compat.v1.nn.dynamic_rnn(
                    cell=single_cell2,
                    inputs=self.outputs1,
                    initial_state=self.init_state,
                    dtype=tf.float32
                )

            #outputs_y
            with tf.compat.v1.variable_scope('output_sy'):
                w_y=tf.compat.v1.get_variable("softmax_w_y",[nh1,ny])   # w_y (300, 2)
                b_y=tf.compat.v1.get_variable("softmax_b_y",[ny])   # b_y (2, )
                outputs1=tf.reshape(self.outputs1,[-1,nh1])         # outputs1 (?, 300)
                sy=tf.compat.v1.nn.xw_plus_b(outputs1,w_y,b_y)      # sy (?, 2)
                self.sy_pred = tf.reshape(tf.argmax(sy, 1), [self.batch_size, -1])  # sy_pred (16, ?)
            #outputs_z
            with tf.compat.v1.variable_scope('output_sz'):
                w_z = tf.get_variable("softmax_w_z", [nh2, nz]) # w_z (300, 5)
                b_z = tf.get_variable("softmax_b_z", [nz])      # b_z (5, )
                outputs2 = tf.reshape(self.outputs2, [-1, nh2]) # outputs2 (?, 300)
                sz = tf.compat.v1.nn.xw_plus_b(outputs2, w_z,b_z)   # sz (?, 5)
                self.sz_pred = tf.reshape(tf.argmax(sz, 1), [self.batch_size, -1])  # sz_pred (16, ?)
            #loss
            with tf.compat.v1.variable_scope('loss'):
                label_y = tf.reshape(self.input_y, [-1])    # label_y (?, )
                # label_y = tf.argmax(self.input_y, 1)
                # loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(sy, label_y)
                loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_y, logits=sy)   # loss1 (?, )
                label_z = tf.reshape(self.input_z, [-1])    # label_z (?, )
                # label_z = tf.argmax(self.input_z, 1)
                # loss2 = tf.nn.softmax_cross_entropy_with_logits(labels=sz, logits=label_z)
                loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_z, logits=sz)   # loss2 (?, )
                self.loss=tf.reduce_sum(0.5*loss1+0.5*loss2)/tf.cast(self.batch_size,tf.float32)

            tvars=tf.compat.v1.trainable_variables()
            grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,tvars),max_gradient_norm)
            optimizer=tf.compat.v1.train.GradientDescentOptimizer(self.lr)
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
 

