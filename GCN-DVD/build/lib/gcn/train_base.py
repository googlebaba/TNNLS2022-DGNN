from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP
from process_data import load_biased_data
checkpt_file = 'pre_trained/mod_cora_baseline{}.ckpt'.format(time.time())
import scipy.sparse as sp
# Set random seed
#seed = 123
#np.random.seed(seed)
#tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay',5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

flags.DEFINE_integer('use_alpha', 1, 'Maximum Chebyshev polynomial degree.')
# Load data
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_biased_data(FLAGS.dataset)

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
import os
'''
bias_label = bias_label.astype(np.float32)
bias_mask = np.ones(train_mask.shape) - val_mask - test_mask
bias_mask = bias_mask.astype(np.bool)
tmp = bias_label[bias_mask]
pos_tmp = np.sum(tmp)
len_tmp = tmp.shape
new_bias_mask = np.zeros(train_mask.shape)
neg_num = 0
for n in range(bias_mask.shape[0]):
    if bias_mask[n]:
        if bias_label[n] == 1:
            new_bias_mask[n] = 1
        elif bias_label[n] == 0 and neg_num <=pos_tmp:
            new_bias_mask[n] = 1
            neg_num += 1
new_bias_mask = new_bias_mask.astype(np.bool)            
tmp = bias_label[new_bias_mask]
pos_tmp = np.sum(tmp)
len_tmp = tmp.shape
print("pos_tmp", pos_tmp)
print("len_tmp", len_tmp)

new_bias_mask = new_bias_mask.astype(np.float32)
'''    
#train_dataset = "dblp"
#test_dataset = "dblp-2008"
#train_A, train_features, y_train_train, y_train_val, train_train_mask, train_val_mask, test_A, test_features, y_test_train, y_test_val, test_train_mask, test_val_mask = load_data(train_dataset, test_dataset)
def compute_P(alpha):
    alpha = alpha/np.sum(alpha)
    alpha = alpha**2
    return (alpha/np.sum(alpha))

def get_index(lst=None, item=''):
        lst = lst.tolist()
        return [index for (index,value) in enumerate(lst) if value == item]
print("adj", adj.shape)
#train_mask_A = train_A.copy()
#train_mask_A[train_val_mask,:] = 0
#train_mask_A[:, train_val_mask] = 0
print(val_mask)
val_mask = val_mask.astype(np.bool)
test_mask = test_mask.astype(np.bool)
val_test_mask = val_mask + test_mask
print(val_test_mask)
train_adj = adj.copy()
print(train_adj.shape)
train_adj[val_test_mask,:] = 0
train_adj[:,val_test_mask] = 0
delete_mask = ((np.sum(train_adj, 1) == 0).reshape(train_mask.shape))

print("train_mask", train_mask)
all_train_mask  = np.ones(train_mask.shape) - delete_mask
#train_mask = all_train_mask * train_mask
train_mask = train_mask.astype(np.bool)
print("train_mask", train_mask)
#print("type", train_adj.todense().dtype)
#train_features = np.eye(train_A.shape[0])
os.environ["CUDA_VISIBLE_DEVICES"]='5'
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#out_features = features
# Some preprocessing
fea_size = features.shape[1]
features = preprocess_features(sp.coo_matrix(features))

if FLAGS.model == 'gcn':
    support_mask = [preprocess_adj(train_adj)]
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    #support = [preprocess_adj(train_A)]  # Not used
    support_mask = [preprocess_adj(train_adj)]
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32, shape=([None])),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
train_size = np.sum(train_mask)
value = np.ones((train_size, 1))
weight_init = tf.constant_initializer(value)
weight = tf.get_variable('weight', shape=[train_size], initializer=weight_init)

train_mask = train_mask.astype(bool)
def run_GCN(lambda1, lambda2):
    # Create model
    model = model_func(placeholders, input_dim=features[2][1],weight=weight,train_size=train_size, label_size =train_mask.shape[0], lambda1= lambda1, lambda2 = lambda2, logging=True)

    # Initialize session
    sess = tf.Session()


    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders, test=False):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)

 #       feed_dict_val.update({placeholders['adjs']: np.array(adj.astype(np.int32))})
        if test:
            outs_val = sess.run([model.lossc, model.accuracy, model.activations[-2]], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)
        outs_val = sess.run([model.lossc, model.accuracy, model.activations[-2], model.hidden_embedding], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test), outs_val[3]


    #saver = tf.train.Saver()
    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0
    # Train model
    feed_dict = construct_feed_dict(features, support_mask, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
    
#        feed_dict.update({placeholders['adjs']: np.array(train_adj.astype(np.float32))})
        # Training step
        
        #outs = sess.run([model.A_embeddings_out, model.adjs_out], feed_dict=feed_dict)
        #print("outs1", outs[0].shape)
        
        #print("outs2", outs[1].shape)
        #outs = sess.run([model.opt_opc, model.lossc, model.accuracy,  model.weight, model.hidden_embedding, model.trans_weight], feed_dict=feed_dict)

        outs = sess.run([model.opt_op, model.loss, model.accuracy, model.weight, model.hidden_embedding], feed_dict=feed_dict)
        #print("alpha", outs[-1])
        #print("weight", outs[-1])
        hidden_embedding = outs[-1]
        hidden_embedding = hidden_embedding[train_mask, :]
#        print("shape", hidden_embedding.shape)
        cor_mat = np.cov(hidden_embedding.T, aweights=outs[-2]*outs[-2])
        n2 = np.linalg.norm(cor_mat,ord=2)
        print("cov**************", n2)


        #outsbias = sess.run([model.opt_op_bias, model.lossb+model.bias_loss, model.accuracy], feed_dict=feed_dict)
        for _ in range(0):
            outsb = sess.run([model.opt_opb, model.lossb, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, emb, duration, _ = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t),"eval time=", "{:.5f}".format(duration))
        '''
        t1 = time.time()
        if acc >= vacc_mx or cost <= vlss_mn:
            if acc >= vacc_mx and cost <= vlss_mn:
                vacc_early_model = acc
                vlss_early_model = cost
                saver.save(sess, checkpt_file)
            vacc_mx = np.max((acc, vacc_mx))
            vlss_mn = np.min((cost, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == FLAGS.early_stopping:
                print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                break
        print("save time=", "{:.5f}".format(time.time() - t1))
        '''
        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    #saver.restore(sess, checkpt_file)

    # Testing
    train_idx = get_index(train_mask, 1)
    test_cost, test_acc, emb, test_duration, test_hidden = evaluate(features, support, y_test, test_mask, placeholders)
    test_hidden = test_hidden[test_mask, :]
    cor_mat = np.cov(test_hidden.T)
    test_n2 = np.linalg.norm(cor_mat,ord=2)
    print("source Test set results:", "cost=","{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    
    return n2, test_n2, test_acc


if __name__ == '__main__':
    n = 10
    para1 = [1]
    para2 = [10]
    best_lambda1 = 0
    best_lambda2 = 0
    best_acc = 0
    best_NI = 0

    best_test_ni = 0
    str1 = str(FLAGS.dataset) + " base"
    for lambda1 in para1:
        for lambda2 in para2:
            NIs = []
            Accs = []
 
            test_corrs = []
            print("best_acc", best_acc)       
            for num in range(n):
                print('n', num)
                print(str1)
                print("best ACC:",best_acc)
                print("lambda1:", lambda1)  
                print("lambda2:", lambda2)  
                NI, test_corr, Acc = run_GCN(lambda1, lambda2)
                NIs.append(NI)
                Accs.append(Acc)
                
                test_corrs.append(test_corr)
            avg_Acc = sum(Accs)/n            
            avg_NI = sum(NIs)/n

            avg_test = sum(test_corrs)/n
            print("avg_Acc", avg_Acc)
            print("avg_NI", avg_NI)
            print("best_acc", best_acc)
            print("best_acc", best_acc)
            if avg_Acc > best_acc:
                best_acc = avg_Acc
                best_NI = avg_NI
                best_lambda1 = lambda1                
                best_lambda2 = lambda2

                best_test_ni = avg_test
    print(Accs)
    print("best ACC:",best_acc)
    print("best train NI:",best_NI)

    print("best test NI:",best_test_ni)
    print("lambda1:", best_lambda1)

    print("lambda2:", best_lambda2)
    print(str1)

