from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP
from process_data import load_biased_data
checkpt_file = 'pre_trained/mod_cora_baseline{}.ckpt'.format(time.time())
import scipy.sparse as sp

import os
# Set random seed
#seed = 123
#np.random.seed(seed)
#tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora_small','cora_medium','cora_big' 'citeseer_small','citeseer_medium', 'citeseer_big', 'pubmed_small', 'pubmed_medium', 'pubmed_big'
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).') #citeseer 20iters, cora,pubmed 10
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

flags.DEFINE_integer('use_DVD', 1, 'whether to use DVD term. 1: GCN-VD/DVD, 0:GCN')

flags.DEFINE_integer('use_alpha', 1, 'weather to differetiate the confounder weights alpha. 1: GCN-DVD, 0:GCN-VD')
flags.DEFINE_float('lambda1', 0, 'lambda1')######### {0.01,0.1,1,10,100}
flags.DEFINE_float('lambda2', 0, 'lambda2')
print("lambda1: ",FLAGS.lambda1)
print("lambda2: ",FLAGS.lambda2)
print("use_alpha: ",FLAGS.use_alpha)
print("early_stopping: ",FLAGS.early_stopping)
# Load data
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_biased_data(FLAGS.dataset)
# use the unbiased dataset 'cora', 'citeseer' 'pubmed'
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

def get_index(lst=None, item=''):
        lst = lst.tolist()
        return [index for (index,value) in enumerate(lst) if value == item]

val_mask = val_mask.astype(np.bool)
test_mask = test_mask.astype(np.bool)
# mask the validation and test nodes
val_test_mask = val_mask + test_mask
train_adj = adj.copy()
train_adj[val_test_mask,:] = 0
train_adj[:,val_test_mask] = 0
delete_mask = ((np.sum(train_adj, 1) == 0).reshape(train_mask.shape))

all_train_mask  = np.ones(train_mask.shape) - delete_mask
train_mask = train_mask.astype(np.bool)
#os.environ["CUDA_VISIBLE_DEVICES"]='0'
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# Some preprocessing
fea_size = features.shape[1]
features = preprocess_features(sp.coo_matrix(features))

if FLAGS.model == 'gcn':
    support_mask = [preprocess_adj(train_adj)]
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    
    support_mask = chebyshev_polynomials(train_adj, FLAGS.max_degree)
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
train_size = np.sum(train_mask)
train_mask = train_mask.astype(bool)
# Create model
placeholders = {
'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
'features': tf.sparse_placeholder(tf.float32),
'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
'labels_mask': tf.placeholder(tf.int32, shape=([None])),
'dropout': tf.placeholder_with_default(0., shape=()),
'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

value = np.ones((train_size, 1))
weight_init = tf.constant_initializer(value)
weight = tf.get_variable('weight', shape=[train_size], initializer=weight_init)
model = model_func(placeholders, input_dim=features[2][1],weight=weight,train_size=train_size, label_size =train_mask.shape[0], lambda1= FLAGS.lambda1, lambda2 = FLAGS.lambda2, logging=True)
# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders, test=False):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)

#       feed_dict_val.update({placeholders['adjs']: np.array(adj.astype(np.int32))})
    outs_val = sess.run([model.lossc, model.accuracy, model.hidden_embedding], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2]


saver = tf.train.Saver()
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
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.weight, model.hidden_embedding], feed_dict=feed_dict)
    last_weight = outs[-2]
    hidden_embedding = outs[-1]
    hidden_embedding = hidden_embedding[train_mask, :]
    cor_mat = np.cov(hidden_embedding.T, aweights=outs[-2]*outs[-2])
    n2 = np.linalg.norm(cor_mat,ord=2)

    if FLAGS.use_DVD:
        for _ in range(1):
            outsb = sess.run([model.opt_opb, model.lossb, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration, _ = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t),"eval time=", "{:.5f}".format(duration))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")


# Testing
train_idx = get_index(train_mask, 1)
test_cost, test_acc, test_duration, test_hidden = evaluate(features, support, y_test, test_mask, placeholders)
if FLAGS.use_DVD == 1:
  f = open("./"+FLAGS.dataset+str(FLAGS.use_alpha)+".txt","a")
  f.write(str(test_acc)+"\n")
  f.close()
else:
  f=open("./" + FLAGS.dataset + "_base.txt","a")
  f.write(str(test_acc)+"\n")
  f.close()
print("Test set results:", "cost=","{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
sess.close()





