import tensorflow as tf

class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    def training(loss, lr, l2_coef):
        # weight decay
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta', 'weight:0']]) * l2_coef

        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        dc_var = [var for var in vars if var.name != 'weight:0']
        # training op
        train_op = opt.minimize(loss+lossL2, var_list = dc_var)
        
        return train_op
    def training_W(loss, weight, lr, l2_coef):
        # weight decay
        vars = tf.trainable_variables()
        #lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
        #                   in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        # optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        # training op
        train_op = opt.minimize(loss, var_list = weight)
        
        return train_op
    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)

##########################
# Adapted from tkipf/gcn #
##########################
    


    def lossb(embedding, alpha, weight, mask, train_size, num, lambda1, lambda2, use_alpha):
        num = num
        loss = 0
        mask = tf.reshape(tf.cast(mask, dtype=tf.bool), [-1])
        embedding = tf.squeeze(embedding)
        embedding = tf.boolean_mask(embedding, mask, axis=0)
        alpha = tf.reshape(alpha, [1,-1])

        new_alpha = alpha/tf.reduce_sum(alpha)

        for p in range(num):
            cfeaturer = embedding
            if p == 0:
                cfeaturer = tf.concat((tf.zeros((train_size,1)), cfeaturer[:,1:num]), axis=1)
            elif p == num-1:                
                cfeaturer = tf.concat((cfeaturer[:,0:num-1], tf.zeros((train_size,1))), axis=1)
            else:                
                cfeaturer = tf.concat((cfeaturer[:,0:p], tf.zeros((train_size, 1)), cfeaturer[:, p+1:num]), axis=1)
            corr_tmp = tf.transpose(tf.matmul(tf.matmul(tf.transpose(embedding[:, p:p+1]), tf.matrix_diag(weight)), cfeaturer)/train_size) - (tf.squeeze(tf.matmul(tf.transpose(embedding[:, p:p+1]),tf.reshape(weight, [-1,1])/train_size))*tf.matmul(tf.transpose(cfeaturer), tf.reshape(weight, [-1,1]))/train_size) 
            if use_alpha:
                loss += tf.reduce_sum(tf.pow(tf.matmul(tf.pow(new_alpha, 1), tf.abs(corr_tmp)), 2))
            else:
                loss += tf.reduce_sum(tf.pow(corr_tmp, 2))
        loss += lambda1*tf.reduce_mean(tf.pow(weight, 2))
        loss += lambda2*tf.pow(tf.reduce_mean(weight)-1, 2)
        return loss


    def masked_softmax_cross_entropy_weight(preds, labels, mask, weight, label_size):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        mask = tf.reshape(tf.cast(mask, dtype=tf.bool), [-1])
        loss = tf.boolean_mask(loss, mask, axis=0)
        loss *= tf.multiply(weight, weight)
        return tf.reduce_sum(loss)/label_size

    def masked_softmax_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_sigmoid_cross_entropy(logits, labels, mask):
        """Softmax cross-entropy loss with masking."""
        labels = tf.cast(labels, dtype=tf.float32)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss=tf.reduce_mean(loss,axis=1)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def masked_accuracy(logits, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def micro_f1(logits, labels, mask):
        """Accuracy with masking."""
        predicted = tf.round(tf.nn.sigmoid(logits))

        # Use integers to avoid any nasty FP behaviour
        predicted = tf.cast(predicted, dtype=tf.int32)
        labels = tf.cast(labels, dtype=tf.int32)
        mask = tf.cast(mask, dtype=tf.int32)

        # expand the mask so that broadcasting works ([nb_nodes, 1])
        mask = tf.expand_dims(mask, -1)
        
        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(predicted * labels * mask)
        tn = tf.count_nonzero((predicted - 1) * (labels - 1) * mask)
        fp = tf.count_nonzero(predicted * (labels - 1) * mask)
        fn = tf.count_nonzero((predicted - 1) * labels * mask)

        # Calculate accuracy, precision, recall and F1 score.
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fmeasure = (2 * precision * recall) / (precision + recall)
        fmeasure = tf.cast(fmeasure, tf.float32)
        return fmeasure
