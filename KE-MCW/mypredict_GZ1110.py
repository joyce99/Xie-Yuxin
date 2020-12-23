import tensorflow as tf
import load
import models.mymodel_mutisize_CNN_LSTM_attention1110 as mymodel
import tools

def batch_putin(train, test, start_num=0, batch_size=16):
    batch = [train[start_num:start_num + batch_size], test[start_num:start_num + batch_size]]
    return batch

def test_batch_putin(test_lex, test_z, start_num=0, batch_size=16):
    test_x = test_lex[start_num:start_num + batch_size]
    test_label_z = test_z[start_num:start_num + batch_size]
    return test_x, test_label_z

def main():
    s = {
        'nh1': 450,
        'nh2': 450,
        'win': 3,
        'emb_dimension': 300,
        'lr': 0.0001,
        'lr_decay': 0.5,  #
        'max_grad_norm': 5,  #
        'seed': 345,  #
        'nepochs': 50,
        'batch_size': 16,
        'keep_prob': 1.0,
        'check_dir': './checkpoints/GZ_mycps_Adam_0.0001_16/kp20k',
        'display_test_per': 1,  #
        'lr_decay_per': 5  #
    }

    # load the dataset
    # data_set_file = 'CNTN/data/inspec_wo_stem/data_set.pkl'
    # emb_file = 'CNTN/data/inspec_wo_stem/embedding.pkl'
    # data_set_file = 'data/ACL2017/krapivin/krapivin_t_a_GZ_data_set.pkl'
    # emb_file = 'data/ACL2017/krapivin/krapivin_t_a_GZ_embedding.pkl'
    data_set_file = 'data/ACL2017/kp20k/kp20k_t_a_data_set.pkl'
    emb_file = 'data/ACL2017/kp20k/kp20k_t_a_embedding.pkl'
    # train_set, test_set, dic, embedding = load.atisfold(data_set_file, emb_file)
    print('loading dataset.....')
    train_set, valid_set, test_set, dic, embedding = load.atisfold_ACL2017(data_set_file, emb_file)
    test_lex, test_y, test_z = test_set

    y_nclasses = 2
    z_nclasses = 5

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)  ###########################################
    with tf.Session(config=config) as sess:
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

        checkpoint_dir = s['check_dir']
        logfile = open(str(s['check_dir']) + '/predict_log_NEW.txt', 'a', encoding='utf-8')
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # print(ckpt.all_model_checkpoint_paths[4])
            print(ckpt.model_checkpoint_path)
            logfile.write(str(ckpt.model_checkpoint_path) + '\n')
            saver.restore(sess, ckpt.model_checkpoint_path)

        def dev_step(cwords):
            feed = {
                my_model.cnn_input_x: cwords,
                my_model.keep_prob: s['keep_prob']
            }
            fetches = my_model.sz_pred
            sz_pred = sess.run(fetches=fetches, feed_dict=feed)
            return sz_pred


        predictions_test = []
        groundtruth_test = []
        start_num = 0
        steps = len(test_lex) // s['batch_size']
        # for batch in tl.iterate.minibatches(test_lex, test_z, batch_size=s['batch_size']):
        print('testing............')
        for step in range(steps):
            # batch = batch_putin(test_lex, test_z, start_num=start_num, batch_size=s['batch_size'])
            x, z = test_batch_putin(test_lex, test_z, start_num=start_num, batch_size=s['batch_size'])
            # x, z = batch
            x = load.pad_sentences(x)
            # x = tools.contextwin_2(x, s['win'])
            predictions_test.extend(dev_step(x))
            groundtruth_test.extend(z)
            start_num += s['batch_size']
            if step % 100 == 0:
                print('tested %d batch......' % (step//100))

        print('dataset: ' + data_set_file)
        logfile.write('dataset: ' + data_set_file + '\n')
        print("测试结果：")
        logfile.write("测试结果：\n")
        res_test = tools.conlleval(predictions_test, groundtruth_test)
        print('all: ', res_test)
        logfile.write('all: ' + str(res_test) + '\n')
        res_test_top5 = tools.conlleval_top(predictions_test, groundtruth_test, 5)
        print('top5: ', res_test_top5)
        logfile.write('top5: ' + str(res_test_top5) + '\n')
        res_test_top10 = tools.conlleval_top(predictions_test, groundtruth_test, 10)
        print('top10: ', res_test_top10)
        logfile.write('top10: ' + str(res_test_top10) + '\n')
        logfile.write('-----------------------------------------------------------------------------------------------------------------------' + '\n')
    logfile.close()

if __name__ == '__main__':
    main()




