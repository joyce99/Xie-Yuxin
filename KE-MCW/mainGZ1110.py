import tensorflow as tf
import time
import os
import load
import models.model1110 as model
import tools

# def batch_putin(train, test, start_num=0, batch_size=16):
#     batch = [train[start_num:start_num+batch_size],test[start_num:start_num+batch_size]]
#     return batch

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
    s={
        'nh1':300,
        'nh2':300,
        'win':3,
        'emb_dimension':300,
        'lr':0.001,
        'lr_decay':0.5,         #
        'max_grad_norm':5,      #
        'seed':345,             #
        'nepochs':45,
        'batch_size':16,
        'keep_prob':0.5,
        'check_dir':'./checkpoints/GZ_EMNLP2016/krapivin_0.001_16',
        'display_test_per':1,   #
        'lr_decay_per':5       #
    }

    data_set_file = 'data/ACL2017/krapivin/krapivin_t_a_GZ_data_set.pkl'
    emb_file = 'data/ACL2017/krapivin/krapivin_t_a_GZ_embedding.pkl'
    print('loading dataset.....')
    # train_set,test_set,dic,embedding = load.atisfold(data_set_file, emb_file)
    train_set, valid_set, test_set, dic, embedding = load.atisfold_ACL2017(data_set_file, emb_file)
    # idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    # idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())

    train_lex, train_y, train_z = train_set
    # train_lex: [[每条tweet的word的idx],[每条tweet的word的idx]], train_y: [[关键词的位置为1]], train_z: [[关键词的位置为0~4(开头、结尾...)]]
    # tr = int(len(train_lex)*0.9)
    # valid_lex, valid_y, valid_z = train_lex[tr:], train_y[tr:], train_z[tr:]
    # train_lex, train_y, train_z = train_lex[:tr], train_y[:tr], train_z[:tr]
    # test_lex,  test_y, test_z  = test_set
    valid_lex, valid_y, valid_z = valid_set
    test_lex, test_y, test_z = test_set
    log_dir = s['check_dir']
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logfile = open(str(s['check_dir']) + '/log.txt', 'a', encoding='utf-8', buffering=1)
    print ('len(train_data) {}'.format(len(train_lex)))
    print ('len(valid_data) {}'.format(len(valid_lex)))
    print ('len(test_data) {}'.format(len(test_lex)))
    logfile.write('len(train_data) {}\n'.format(len(train_lex)))
    logfile.write('len(valid_data) {}\n'.format(len(valid_lex)))
    logfile.write('len(test_data) {}\n'.format(len(test_lex)))
    vocab = set(dic['words2idx'].keys())
    vocsize = len(vocab)
    print ('len(vocab) {}'.format(vocsize))
    print ("Train started!")
    logfile.write('len(vocab) {}\n'.format(vocsize))
    logfile.write("Train started!\n")
    y_nclasses = 2
    z_nclasses = 5

    nsentences = len(train_lex)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True)  ###########################################
    with tf.Session(config=config) as sess:  #####################################
        rnn=model.Model(
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
    #     my_model = mymodel.myModel(
    #         # nh1=s['nh1'],
    #         # nh2=s['nh2'],
    #         # ny=y_nclasses,
    #         # nz=z_nclasses,
    #         de=s['emb_dimension'],
    #         lr=s['lr'],
    #         lr_decay=s['lr_decay'],
    #         embedding=embedding,
    #         max_gradient_norm=s['max_grad_norm'],
    #         keep_prob=s['keep_prob'],
    #         model_cell='lstm'
    #     )

        # 保存模型
        checkpoint_dir=s['check_dir']
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        checkpoint_prefix=os.path.join(checkpoint_dir,'model')

        def train_step(cwords,label_y,label_z):
            feed={
                rnn.input_x:cwords,
                rnn.input_y:label_y,
                rnn.input_z:label_z,
                rnn.keep_prob:s['keep_prob']
                # rnn.batch_size:s['batch_size']
            }
            fetches=[rnn.loss,rnn.train_op]
            loss,_=sess.run(fetches=fetches,feed_dict=feed)
            # _,Loss = sess.run(fetches=fetches, feed_dict=feed)
            return loss

        def dev_step(cwords):
            feed={
                rnn.input_x:cwords,
                rnn.keep_prob: 1.0
                # rnn.keep_prob:1.0,
                # rnn.batch_size:s['batch_size']
            }
            fetches=rnn.sz_pred
            sz_pred=sess.run(fetches=fetches,feed_dict=feed)
            return sz_pred

        saver=tf.train.Saver(tf.all_variables(), max_to_keep=2)
        sess.run(tf.global_variables_initializer())

        best_f=-1
        best_e=0
        test_best_f=-1
        test_best_e=0
        best_res=None
        test_best_res=None
        for e in range(s['nepochs']):
            tools.shuffle([train_lex,train_y,train_z],s['seed'])
            t_start=time.time()
            start_num = 0
            # for step,batch in enumerate(tl.iterate.minibatches(train_lex,list(zip(train_y,train_z)),batch_size=s['batch_size'])):
            # for step, batch in enumerate(batch_putin(train_lex, list(zip(train_y, train_z)), start_num=start_num, batch_size=s['batch_size'])):
            steps = len(train_lex) // s['batch_size']
            for step in range(steps):
                # batch = batch_putin(train_lex,list(zip(train_y, train_z)), start_num=start_num, batch_size=s['batch_size'])
                # input_x,target=batch
                # label_y,label_z=list(zip(*target))
                input_x, label_y, label_z = train_batch_putin(train_lex, train_y, train_z, start_num=start_num,
                                                              batch_size=s['batch_size'])
                input_x=load.pad_sentences(input_x)
                label_y=load.pad_sentences(label_y)
                label_z=load.pad_sentences(label_z)
                cwords=tools.contextwin_2(input_x,s['win'])
                # cwords = input_x
                loss=train_step(cwords,label_y,label_z)
                start_num += s['batch_size']
                print('loss %.6f' % loss,
                      ' [learning] epoch %i>> %2.2f%%' % (e, s['batch_size'] * step * 100. / nsentences),
                      'completed in %.2f (sec) <<\r' % (time.time() - t_start))
                if step % 1000 == 0:
                    logfile.write('loss %.6f' % loss)
                    logfile.write(' [learning] epoch %i>> %2.2f%%' % (e, s['batch_size'] * step * 100. / nsentences))
                    logfile.write('completed in %.2f (sec) <<\n' % (time.time() - t_start))
                # sys.stdout.flush())

            #VALID
            if e >= 0:
                print('Validing..............')
                predictions_valid=[]
                predictions_test=[]
                groundtruth_valid=[]
                groundtruth_test=[]
                start_num = 0
                steps = len(valid_lex) // s['batch_size']
                # for batch in  tl.iterate.minibatches(valid_lex,valid_z,batch_size=s['batch_size']):
                for step in range(steps):
                    # batch = batch_putin(valid_lex, valid_z, start_num=start_num, batch_size=s['batch_size'])
                    # x,z=batch
                    x, z = test_batch_putin(valid_lex, valid_z, start_num=start_num, batch_size=s['batch_size'])
                    x=load.pad_sentences(x)
                    x=tools.contextwin_2(x,s['win'])
                    predictions_valid.extend(dev_step(x))
                    groundtruth_valid.extend(z)
                    start_num += s['batch_size']

                res_valid=tools.conlleval(predictions_valid,groundtruth_valid)
                del predictions_valid
                del groundtruth_valid
                if res_valid['f']>best_f:
                    best_f=res_valid['f']
                    best_e=e
                    best_res=res_valid
                    print ('\nVALID new best:',res_valid)
                    logfile.write('\nVALID new best: ' + str(res_valid))
                    path = saver.save(sess=sess, save_path=checkpoint_prefix, global_step=e)
                    print ("Save model checkpoint to {}".format(path))
                    logfile.write("\nSave model checkpoint to {}\n".format(path))
                else:
                    print ('\nVALID new curr:',res_valid)
                    logfile.write('\nVALID new curr: ' + str(res_valid))

                #TEST
                print('Testing..............')
                start_num = 0
                steps = len(test_lex) // s['batch_size']
                if e%s['display_test_per']==0:
                    # for batch in tl.iterate.minibatches(test_lex, test_z, batch_size=s['batch_size']):
                    for step in range(steps):
                        # batch = batch_putin(test_lex, test_z, start_num=start_num, batch_size=s['batch_size'])
                        # x,z = batch
                        x, z = test_batch_putin(test_lex, test_z, start_num=start_num, batch_size=s['batch_size'])
                        x = load.pad_sentences(x)
                        x = tools.contextwin_2(x, s['win'])
                        predictions_test.extend(dev_step(x))
                        groundtruth_test.extend(z)
                        start_num += s['batch_size']


                    res_test = tools.conlleval(predictions_test, groundtruth_test)

                    if res_test['f'] > test_best_f:
                        test_best_f = res_test['f']
                        test_best_e=e
                        test_best_res=res_test
                        print ('TEST new best:',res_test)
                        logfile.write('\nTEST new best: ' + str(res_test))
                    else:
                        print ('TEST new curr:',res_test)
                        logfile.write('\nTEST new curr: ' + str(res_test))

                # learning rate decay if no improvement in 10 epochs
                if e-best_e>s['lr_decay_per']:
                    sess.run(fetches=rnn.learning_rate_decay_op)
                lr=sess.run(fetches=rnn.lr)
                print ('learning rate:%f' % lr)
                logfile.write('\nlearning rate:%f\n' % lr)
                if lr<1e-6:break

        print ("Train finished!")
        print ('Valid Best Result: epoch %d:  ' % (best_e),best_res)
        print ('Test Best Result: epoch %d:  ' %(test_best_e),test_best_res)
        logfile.write("Train finished!\n")
        logfile.write('Valid Best Result: epoch %d:   ' % (best_e) + str(best_res))
        logfile.write('\nTest Best Result: epoch %d:   ' % (test_best_e) + str(test_best_res))
        logfile.close()

if __name__ == '__main__':
    main()
