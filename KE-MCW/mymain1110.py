import tensorflow as tf
import time
import os
import load
import models.mymodel_mutisize_CNN_LSTM_attention1110 as mymodel
import tools


def train_batch_putin(train_lex, train_y, train_z, start_num=0, batch_size=16):
    # batch = [train_lex[start_num:start_num + batch_size], train_y[start_num:start_num + batch_size], train_z[start_num:start_num + batch_size]]
    # return batch
    input_x = train_lex[start_num:start_num + batch_size]
    label_y = train_y[start_num:start_num + batch_size]
    label_z = train_z[start_num:start_num + batch_size]
    return input_x, label_y, label_z

def test_batch_putin(test_lex, test_z, start_num=0, batch_size=16):
    test_x = test_lex[start_num:start_num + batch_size]
    test_label_z = test_z[start_num:start_num + batch_size]
    return test_x, test_label_z


def main():
    s = {
        'nh1': 450, # 第1层LSTM的隐藏单元数
        'nh2': 450, # 第2层LSTM的隐藏单元数
        'emb_dimension': 300,   # 词向量维度
        'lr': 0.0001,  # 初始学习率
        'lr_decay': 0.5,  # 学习率衰减率
        'lr_decay_per': 5,  # 如果训练5次以后准确率没有上升，则衰减学习率为原来的0.5倍
        'nepochs': 50,  # 总共迭代50个epoch
        'batch_size': 16,   # batch_size=16
        'keep_prob': 0.5,   # drop out 概率
        'check_dir': './checkpoints/kp20k_mycps_multisize_CNN_LSTM_attention_Adam_0.0001_16_again', # 模型保存地址
        'max_grad_norm': 5,  #
        'seed': 345,  #
        'display_test_per': 1,  #
    }

    data_set_file ='data/ACL2017/kp20k/kp20k_t_a_allwords_data_set.pkl'
    emb_file = 'data/ACL2017/ACL2017_t_a_embedding.pkl'
    # train_set, test_set, dic, embedding = load.atisfold(data_set_file, emb_file)
    print('loading dataset.....')
    train_set, valid_set, test_set, dic, embedding = load.atisfold_ACL2017(data_set_file, emb_file)

    train_lex, train_y, train_z = train_set
    # train_lex: [[每条tweet的word的idx],[每条tweet的word的idx]], train_y: [[关键词的位置为1]], train_z: [[关键词的位置为0~4(开头、结尾...)]]
    # tr = int(len(train_lex) * 0.9)
    # valid_lex, valid_y, valid_z = train_lex[tr:], train_y[tr:], train_z[tr:]    ################
    # train_lex, train_y, train_z = train_lex[:tr], train_y[:tr], train_z[:tr]
    valid_lex, valid_y, valid_z = valid_set
    test_lex, test_y, test_z = test_set
    log_dir = s['check_dir']
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logfile = open(str(s['check_dir']) + '/log.txt', 'a', encoding='utf-8', buffering=1)
    print('len(train_data) {}'.format(len(train_lex)))
    print('len(valid_data) {}'.format(len(valid_lex)))
    print('len(test_data) {}'.format(len(test_lex)))
    logfile.write('len(train_data) {}\n'.format(len(train_lex)))
    logfile.write('len(valid_data) {}\n'.format(len(valid_lex)))
    logfile.write('len(test_data) {}\n'.format(len(test_lex)))
    vocab = set(dic['words2idx'].keys())
    vocsize = len(vocab)
    print('len(vocab) {}'.format(vocsize))
    print("Train started!")
    logfile.write('len(vocab) {}\n'.format(vocsize))
    logfile.write("Train started!\n")
    y_nclasses = 2
    z_nclasses = 5

    nsentences = len(train_lex)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True, allow_soft_placement=True)###########################################
    with tf.Session(config=config) as sess:#####################################
        my_model = mymodel.myModel(
            nh1=s['nh1'],
            nh2=s['nh2'],
            ny=y_nclasses,
            nz=z_nclasses,
            de=s['emb_dimension'],
            lr=s['lr'],
            lr_decay=s['lr_decay'],
            embedding=embedding,
            max_gradient_norm=s['max_grad_norm'],
            batch_size=s['batch_size'],
            rnn_model_cell='lstm'
        )

        # 保存模型
        checkpoint_dir=s['check_dir']
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_prefix=os.path.join(checkpoint_dir,'model')

        def train_step(cwords, label_y, label_z):
            feed = {
                my_model.cnn_input_x: cwords,
                my_model.rnn_input_y: label_y,
                my_model.rnn_input_z: label_z,
                my_model.keep_prob: s['keep_prob']
            }
            fetches = [my_model.loss, my_model.train_op]
            loss, _ = sess.run(fetches=fetches, feed_dict=feed)
            return loss

        def dev_step(cwords):
            feed={
                my_model.cnn_input_x:cwords,
                my_model.keep_prob: 1.0
            }
            fetches=my_model.sz_pred
            sz_pred=sess.run(fetches=fetches,feed_dict=feed)
            return sz_pred

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=3)
        sess.run(tf.global_variables_initializer())
        best_f = -1
        best_e = 0
        decay_e = 0
        test_best_f = -1
        test_best_e = 0
        best_res = None
        test_best_res = None
        for e in range(s['nepochs']):
            tools.shuffle([train_lex, train_y, train_z], s['seed'])
            t_start = time.time()
            start_num = 0
            steps = len(train_lex) // s['batch_size']
            for step in range(steps):       ##################################################
                input_x, label_y, label_z = train_batch_putin(train_lex, train_y, train_z, start_num=start_num, batch_size=s['batch_size'])
                # input_x, label_y, label_z = batch
                # label_y, label_z = list(zip(*target))
                input_x = load.pad_sentences(input_x)
                label_y = load.pad_sentences(label_y)
                label_z = load.pad_sentences(label_z)
                loss = train_step(input_x, label_y, label_z)
                start_num += s['batch_size']
                print('loss %.6f' % loss,
                      ' [learning] epoch %i>> %2.2f%%' % (e, s['batch_size'] * step * 100. / nsentences),
                      'completed in %.2f (sec) <<\r' % (time.time() - t_start))
                if step % 1000 == 0:
                    logfile.write('loss %.6f' % loss)
                    logfile.write(' [learning] epoch %i>> %2.2f%%' % (e, s['batch_size'] * step * 100. / nsentences))
                    logfile.write('completed in %.2f (sec) <<\n' % (time.time() - t_start))

            # VALID
            if e >= 0:
                print('Validing..............')
                predictions_valid = []
                predictions_test = []
                groundtruth_valid = []
                groundtruth_test = []
                start_num = 0
                steps = len(valid_lex) // s['batch_size']
                for step in range(steps):
                    x, z = test_batch_putin(valid_lex, valid_z, start_num=start_num, batch_size=s['batch_size'])
                    # x, z = batch
                    x = load.pad_sentences(x)
                    predictions_valid.extend(dev_step(x))
                    groundtruth_valid.extend(z)
                    start_num += s['batch_size']

                res_valid = tools.conlleval(predictions_valid, groundtruth_valid)
                del predictions_valid
                del groundtruth_valid
                if res_valid['f'] > best_f:
                    best_f = res_valid['f']
                    best_e = e
                    decay_e = e
                    best_res = res_valid
                    print('\nVALID new best:', res_valid)
                    logfile.write('\nVALID new best: ' + str(res_valid))
                    path = saver.save(sess=sess, save_path=checkpoint_prefix, global_step=e)
                    print("Save model checkpoint to {}".format(path))
                    logfile.write("\nSave model checkpoint to {}\n".format(path))
                else:
                    print('\nVALID new curr:', res_valid)
                    logfile.write('\nVALID new curr: ' + str(res_valid))

                # TEST
                print('Testing..............')
                start_num = 0
                steps = len(test_lex) // s['batch_size']
                if e % s['display_test_per'] == 0:
                    for step in range(steps):
                        x, z = test_batch_putin(test_lex, test_z, start_num=start_num, batch_size=s['batch_size'])
                        # x, z = batch
                        x = load.pad_sentences(x)
                        predictions_test.extend(dev_step(x))
                        groundtruth_test.extend(z)
                        start_num += s['batch_size']

                    res_test = tools.conlleval(predictions_test, groundtruth_test)
                    del predictions_test
                    del groundtruth_test
                    if res_test['f'] > test_best_f:
                        test_best_f = res_test['f']
                        test_best_e = e
                        test_best_res = res_test
                        print('TEST new best:', res_test)
                        logfile.write('\nTEST new best: ' + str(res_test))
                    else:
                        print('TEST new curr:', res_test)
                        logfile.write('\nTEST new curr: ' + str(res_test))

                # learning rate decay if no improvement in 10 epochs
                if e - best_e > s['lr_decay_per'] and e - decay_e > s['lr_decay_per']:
                    sess.run(fetches=my_model.learning_rate_decay_op)
                    decay_e = e
                lr = sess.run(fetches=my_model.lr)
                print('learning rate:%f' % lr)
                logfile.write('\nlearning rate:%f\n' % lr)
                if lr < 1e-6: break

        print("Train finished!")
        print('Valid Best Result: epoch %d:  ' % (best_e), best_res)
        print('Test Best Result: epoch %d:  ' % (test_best_e), test_best_res)
        logfile.write("Train finished!\n")
        logfile.write('Valid Best Result: epoch %d:   ' % (best_e) + str(best_res))
        logfile.write('\nTest Best Result: epoch %d:   ' % (test_best_e) + str(test_best_res))
        logfile.close()


if __name__ == '__main__':
    main()
