import tensorflow as tf
import load
import models.model as model
import tools

def batch_putin(train, test, start_num=0, batch_size=16):
    batch = [train[start_num:start_num+batch_size],test[start_num:start_num+batch_size]]
    return batch

def test_batch_putin(test_lex, test_z, start_num=0, batch_size=16):
    test_x = test_lex[start_num:start_num + batch_size]
    test_label_z = test_z[start_num:start_num + batch_size]
    return test_x, test_label_z

def main():
    s={
        'nh1':300,
        'nh2':300,
        'win':3,
        'emb_dimension':300,
        'lr':0.001,
        'lr_decay':0.5,
        'max_grad_norm':5,
        'seed':345,
        'nepochs':50,
        'batch_size':16,
        'keep_prob':1.0,
        'check_dir':'./checkpoints/GZ_EMNLP2016/semeval_0.001_16',
        'display_test_per':1,
        'lr_decay_per':5
    }

    
    # load the dataset
    #data_set_file = 'data/ACL2017/inspec/inspec_t_a_GZ_data_set.pkl'
    #emb_file = 'data/ACL2017/inspec/inspec_t_a_GZ_embedding.pkl'
    # data_set_file = 'data/ACL2017/krapivin/krapivin_t_a_allwords_data_set.pkl'
    # emb_file = 'data/ACL2017/ACL2017_t_a_embedding.pkl'
    # data_set_file = 'data/ACL2017/krapivin/krapivin_t_a_GZ_data_set.pkl'
    # emb_file = 'data/ACL2017/krapivin/krapivin_t_a_GZ_embedding.pkl'
    #data_set_file = 'data/ACL2017/kp20k/kp20k_t_a_allwords_data_set.pkl'
    #emb_file = 'data/ACL2017/kp20k/ACL2017_t_a_embedding.pkl'
    data_set_file = 'data/ACL2017/semeval/semeval_t_a_GZ_data_set.pkl'
    emb_file = 'data/ACL2017/semeval/semeval_t_a_GZ_embedding.pkl'
    # train_set, test_set, dic, embedding = load.atisfold(data_set_file, emb_file)
    #data_set_file = 'data/ACL2017/nus/nus_t_a_GZ_data_set.pkl'
    #emb_file = 'data/ACL2017/nus/nus_t_a_GZ_embedding.pkl'
    # train_set, test_set, dic, embedding = load.atisfold(data_set_file, emb_file)
    print('loading dataset.....')
    train_set, valid_set, test_set, dic, embedding = load.atisfold_ACL2017(data_set_file, emb_file)
    # idx2label = dict((k,v) for v,k in dic['labels2idx'].items())
    # idx2word  = dict((k,v) for v,k in dic['words2idx'].items())

    # vocab = set(dic['words2idx'].keys())
    # vocsize = len(vocab)
    test_lex,  test_y, test_z  = test_set
    # test_lex = test_lex[:1000]
    # test_y = test_y[:1000]
    # test_z = test_z[:1000]

    y_nclasses = 2
    z_nclasses = 5

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)
    with tf.Session(config=config) as sess:

        rnn = model.Model(
            nh1=s['nh1'],
            nh2=s['nh2'],
            ny=y_nclasses,
            nz=z_nclasses,
            de=s['emb_dimension'],
            cs=s['win'],
            lr=s['lr'],
            lr_decay=s['lr_decay'],
            embedding=embedding,
            max_gradient_norm=s['max_grad_norm'],
            batch_size=s['batch_size'],
            model_cell='lstm'
        )

        checkpoint_dir = s['check_dir']
        logfile = open(str(s['check_dir']) + '/predict_log_NEW.txt', 'a', encoding='utf-8')
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            logfile.write(str(ckpt.model_checkpoint_path) + '\n')
            saver.restore(sess, ckpt.model_checkpoint_path)

        def dev_step(cwords):
            feed={
                rnn.input_x:cwords,
                rnn.keep_prob: s['keep_prob'],
                # rnn.batch_size:s['batch_size']
            }
            fetches=rnn.sz_pred
            sz_pred=sess.run(fetches=fetches,feed_dict=feed)
            return sz_pred

        predictions_test=[]
        groundtruth_test=[]
        start_num = 0
        steps = len(test_lex) // s['batch_size']
        # for batch in tl.iterate.minibatches(test_lex, test_z, batch_size=s['batch_size']):
        print('testing............')
        for step in range(steps):
            # batch = batch_putin(test_lex, test_z, start_num=start_num, batch_size=s['batch_size'])
            # x, z = batch
            x, z = test_batch_putin(test_lex, test_z, start_num=start_num, batch_size=s['batch_size'])
            x = load.pad_sentences(x)
            x = tools.contextwin_2(x, s['win'])
            predictions_test.extend(dev_step(x))
            groundtruth_test.extend(z)
            start_num += s['batch_size']
            if step % 100 == 0:
                print('tested %d batch......' % step)

        print('dataset: ' + data_set_file)
        logfile.write('dataset: ' + data_set_file + '\n')
        print("result:")
        logfile.write("result:\n")
        # res_test = tools.conlleval(predictions_test, groundtruth_test)
        res_test = tools.conlleval(predictions_test, groundtruth_test)
        print('all: ', res_test)
        logfile.write('all: ' + str(res_test) + '\n')
        res_test_top5 = tools.conlleval_top(predictions_test, groundtruth_test, 5)
        print('top5: ', res_test_top5)
        logfile.write('top5: ' + str(res_test_top5) + '\n')
        res_test_top10 = tools.conlleval_top(predictions_test, groundtruth_test, 10)
        print('top10: ', res_test_top10)
        logfile.write('top10: ' + str(res_test_top10) + '\n')
        logfile.write('-----------------------------------------------------------------------------------------------------------------------' +'\n')
    logfile.close()

if __name__ == '__main__':
    main()




