from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP
from process_data import load_biased_data
checkpt_file = 'pre_trained/mod_cora_baseline.ckpt'
import scipy.sparse as sp
# Set random seed
#seed = 123
#np.random.seed(seed)
#tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
import os


def get_index(lst=None, item=''):
        lst = lst.tolist()
        return [index for (index,value) in enumerate(lst) if value == item]

train_val_idx = get_index(train_val_mask, 1)
test_val_idx = get_index(test_val_mask, 1)
source_val_mask = np.zeros(train_val_mask.shape)
source_test_mask = np.zeros(train_val_mask.shape)
source_val_mask[train_val_idx[:500]] = 1
source_test_mask[train_val_idx[500:]] = 1
print(source_val_mask)
source_val_mask = source_val_mask.astype(bool)
source_test_mask = source_test_mask.astype(bool)

target_val_mask = np.zeros(test_val_mask.shape)
target_test_mask = np.zeros(test_val_mask.shape)
target_val_mask[test_val_idx[:500]] = 1
target_test_mask[test_val_idx[500:]] = 1

target_val_mask = target_val_mask.astype(bool)
target_test_mask = target_test_mask.astype(bool)


#train_mask_A = train_A.copy()
#train_mask_A[train_val_mask,:] = 0
#train_mask_A[:, train_val_mask] = 0

y_train = y_train_train
y_val = y_train_val
y_test = y_test_val
train_mask = train_train_mask
val_mask = train_val_mask
test_mask = np.ones(test_A.shape[0])

#train_features = np.eye(train_A.shape[0])
os.environ["CUDA_VISIBLE_DEVICES"]=str(3)
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#out_features = features
# Some preprocessing
features = preprocess_features(sp.coo_matrix(train_features))
features_test = preprocess_features(sp.coo_matrix(test_features))

if FLAGS.model == 'gcn':
    support_mask = [preprocess_adj(train_A)]
    support = [preprocess_adj(train_A)]
    support_test = [preprocess_adj(test_A)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(train_A)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32, shape=(None)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
train_size = np.sum(train_mask)
value = np.ones((train_size, 1))
weight_init = tf.constant_initializer(value)
weight = tf.get_variable('weight', shape=[train_size], initializer=weight_init)

def run_GCN():
    # Create model
    model = model_func(placeholders, input_dim=features[2][1],weight=weight,train_size=train_size, logging=True)

    # Initialize session
    sess = tf.Session()


    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders, test=False):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        if test:
            outs_val = sess.run([model.lossc, model.accuracy, model.activations[-2]], feed_dict=feed_dict_val)
            return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)
        outs_val = sess.run([model.lossc, model.accuracy, model.activations[-2]], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)


    saver = tf.train.Saver()
    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []
    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0


    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_opc, model.lossc, model.accuracy], feed_dict=feed_dict)
        #for _ in range(5):
        #    outsb = sess.run([model.opt_opb, model.lossb, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, emb, duration = evaluate(features, support, y_val, source_val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
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


        #if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        #    print("Early stopping...")
        #    break

    print("Optimization Finished!")

    saver.restore(sess, checkpt_file)

    # Testing
    train_idx = get_index(train_mask, 1)

    source_test_idx = get_index(source_test_mask, 1)
    target_test_idx = get_index(target_test_mask, 1)

    test_cost, test_acc, emb, test_duration = evaluate(features, support, y_train, train_mask, placeholders)
    train_embeddings = []
    for n in train_idx:
        train_embeddings.append(emb[n,:])
    train_embeddings = np.vstack(train_embeddings)
    np.save("train.npy", train_embeddings)

    test_cost, test_acc, emb, test_duration = evaluate(features, support, y_val, source_test_mask, placeholders)
    test_embeddings1 = []
    for n in source_test_idx:
        test_embeddings1.append(emb[n,:])
    test_embeddings1 = np.vstack(test_embeddings1)
    print("source Test set results:", "cost=","{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))


    test_cost, test_acc, emb, test_duration = evaluate(features_test, support_test, y_test, target_test_mask, placeholders)
    test_embeddings = []
    for n in target_test_idx:
        test_embeddings.append(emb[n,:])
    test_embeddings = np.vstack(test_embeddings)
    np.save("test.npy", test_embeddings)
    print("Test set results:", "cost=","{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    NI = np.linalg.norm((np.mean(train_embeddings, axis=0)-np.mean(test_embeddings1, axis=0))/np.std(np.concatenate((train_embeddings, test_embeddings1), axis=0), axis=0), ord=2)
    print("self NI", NI)
    NI = np.linalg.norm((np.mean(train_embeddings, axis=0)-np.mean(test_embeddings, axis=0))/np.std(np.concatenate((train_embeddings, test_embeddings), axis=0), axis=0), ord=2)
    print("transfer NI", NI)
    return NI, test_acc


if __name__ == '__main__':
    NIs = []
    Accs = []
    n = 10
    for _ in range(n):
        NI, Acc = run_GCN()
        NIs.append(NI)
        Accs.append(Acc)
    print(Accs)
    print("ACC:",sum(Accs)/n)
    print("NI:",sum(NIs)/n)
