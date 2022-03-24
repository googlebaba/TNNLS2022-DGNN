import time
import numpy as np
import tensorflow as tf
import scipy.io as sio

import scipy.sparse as sp
from models import GAT
from utils import process
import os
from process_data import load_biased_data 
checkpt_file = 'pre_trained/cora/mod_cora_{}.ckpt'.format(time.time())
# dataset: cora_small, cora_medium, cora_big, citeseer_small, citeseer_medium, citeseer_big, pubmed_small, pubmed_medium, pubmed_big
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora_big', 'Dataset string.')
flags.DEFINE_integer('use_DVD', 1, 'whether to use DVD term. 1: GCN-VD/DVD, 0:GCN')

flags.DEFINE_integer('use_alpha', 1, 'weather to differetiate the confounder weights alpha. 1: GCN-DVD, 0:GCN-VD')
flags.DEFINE_float('lambda1', 0, 'lambda1')######### {0.01,0.1,1,10,100}
flags.DEFINE_float('lambda2', 0, 'lambda2')

# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

use_DVD = FLAGS.use_DVD    # whether to use the VD or DVD term  1:GAT-VD/DVD 0: GAT
use_alpha = FLAGS.use_alpha  # whether to use confounder weight   1:DVD 0:VD
weight_round = 1
lambda1 = FLAGS.lambda1               # lambda1
lambda2 = FLAGS.lambda2                # lambda2
print('Dataset: ' + FLAGS.dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

#os.environ["CUDA_VISIBLE_DEVICES"]='2'
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(FLAGS.dataset) #for loading baseline
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_biased_data(FLAGS.dataset)
features, spars = process.preprocess_features(sp.coo_matrix(features))
adj = adj.todense()
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]
adj_train = adj.copy()
val_test_mask = val_mask + test_mask
adj_train[val_test_mask.tolist(), :] = 0
adj_train[:,val_test_mask.tolist()] = 0

features = features[np.newaxis]
adj = adj[np.newaxis]
adj_train = adj_train[np.newaxis]

y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]
train_size = np.sum(train_mask)

biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

biases_train = process.adj_to_bias(adj_train, [nb_nodes], nhood=1)

with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
        bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=())

        P_in = tf.placeholder(dtype=tf.float32, shape=(64))
        is_train = tf.placeholder(dtype=tf.bool, shape=())
    value = np.ones((train_size, 1))
    weight_init = tf.constant_initializer(value)
    weight = tf.get_variable('weight', shape=[train_size], initializer=weight_init)


    logits, final_embedding, alpha  = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                attn_drop, ffd_drop,
                                bias_mat=bias_in,
                                hid_units=hid_units, n_heads=n_heads,
                                residual=residual, activation=nonlinearity)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy_weight(log_resh, lab_resh, msk_resh, weight,train_mask.shape[1])

    weight_mean, alpha = tf.nn.moments(alpha, axes=-1)
    #loss += 0.001*tf.reduce_sum(alpha)
    lossc = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)

    lossb = model.lossb(final_embedding, alpha, tf.multiply(weight, weight), msk_resh, train_size, final_embedding.shape.as_list()[2], lambda1, lambda2, use_alpha)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)


    train_W = model.training_W(lossb, weight, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

   
        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
                    feed_dict={
                        ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                        bias_in: biases_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                        msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                        is_train: True,
                        attn_drop: 0.6, ffd_drop: 0.6})
                if use_DVD:
                    for _ in range(weight_round):						
                        _, _, _, weight_OUT = sess.run([train_W, lossb, accuracy, weight],
                            feed_dict={
                                ftr_in: features[tr_step*batch_size:(tr_step+1)*batch_size],
                                bias_in: biases_train[tr_step*batch_size:(tr_step+1)*batch_size],
                                lbl_in: y_train[tr_step*batch_size:(tr_step+1)*batch_size],
                                msk_in: train_mask[tr_step*batch_size:(tr_step+1)*batch_size],
                                is_train: True,
                                attn_drop: 0.6, ffd_drop: 0.6})
                #    print("weight", weight_OUT)
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, acc_vl = sess.run([lossc, accuracy],
                    feed_dict={
                        ftr_in: features[vl_step*batch_size:(vl_step+1)*batch_size],
                        bias_in: biases[vl_step*batch_size:(vl_step+1)*batch_size],
                        lbl_in: y_val[vl_step*batch_size:(vl_step+1)*batch_size],
                        msk_in: val_mask[vl_step*batch_size:(vl_step+1)*batch_size],
                        is_train: False,
                        attn_drop: 0.0, ffd_drop: 0.0})
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print('Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                    (train_loss_avg/tr_step, train_acc_avg/tr_step,
                    val_loss_avg/vl_step, val_acc_avg/vl_step))

            if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                    vacc_early_model = val_acc_avg/vl_step
                    vlss_early_model = val_loss_avg/vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                    print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0


        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run([lossc, accuracy],
                feed_dict={
                    ftr_in: features[ts_step*batch_size:(ts_step+1)*batch_size],
                    bias_in: biases[ts_step*batch_size:(ts_step+1)*batch_size],
                    lbl_in: y_test[ts_step*batch_size:(ts_step+1)*batch_size],
                    msk_in: test_mask[ts_step*batch_size:(ts_step+1)*batch_size],
                    is_train: False,
                    attn_drop: 0.0, ffd_drop: 0.0})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1
        f = open("./"+FLAGS.dataset+str(FLAGS.use_alpha)+".txt","a")
        f.write(str(ts_acc/ts_step)+"\n")
        f.close()
        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)

        sess.close()


