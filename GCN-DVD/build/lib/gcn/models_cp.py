from gcn.layers import *
from gcn.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.lossb = 0
        self.lossb1 = 0
        self.lossc = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        n = 0
        def tf_cov(x):
            mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
            cov_xx = tf.matmul(tf.transpose(x-mean_x), x-mean_x)/tf.cast(tf.shape(x)[0]-1, tf.float32)
            return cov_xx

        for layer in self.layers:
            #if n != 0:
            #    self.activations[-1] = tf.layers.batch_normalization(self.activations[-1])
            print(n)
            n += 1
            hidden = layer(self.activations[-1])                
            self.activations.append(hidden)
            
        self.outputs = self.activations[-1]

        self.hidden_embedding = self.activations[-2]
        #self.hidden_embedding = self.layers[-1].pre_sup
        self.trans_weight1 = self.layers[-1].trans_weight
        weight_mean, self.trans_weight = tf.nn.moments(self.trans_weight1, axes=1)
        self.alpha = glorot([self.hidden_embedding.shape.as_list()[-1], 1], name='alpha')
        
        #bias = zeros([1], name='bias')
        #self.bias_logits = tf.sigmoid(tf.matmul(self.hidden_embedding, self.alpha)+bias)
        #one = tf.ones_like(self.bias_logits)
        #zero = tf.zeros_like(self.bias_logits)
        #self.bias_logits_acc = tf.where(self.bias_logits <0.5, x=zero, y=one)
        '''        
        self.A_embeddings = tf.layers.dense(inputs=self.hidden_embedding, units=64, activation=tf.nn.relu)
        self.X_embeddings = tf.layers.dense(inputs=self.hidden_embedding, units=64, activation=tf.nn.relu)
        A_mean = tf.reduce_mean(self.A_embeddings, axis=0)
        X_mean = tf.reduce_mean(self.X_embeddings, axis=0)
        print("A_mean", A_mean)
        print("X_mean", X_mean)
        w1 = tf.exp(attention("A", A_mean))
        w2 = tf.exp(attention("X", X_mean))
        a1 = [w1/(w1+w2), w2/(w1+w2)]
        
        mix_embeddings = w1/(w1+w2)*A_embeddings + w2/(w1+w2)*X_embeddings
        '''
    #    C = tf_cov(self.hidden_embedding)
    #    loss_cov = tf.reduce_mean(tf.pow(C,2))  - tf.reduce_mean(tf.pow(tf.diag_part(C), 2))
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        vars_all = tf.trainable_variables()
        print("self.weight.name", self.weight.name)
        self.vars_list = [var for var in vars_all if var.name!=self.weight.name]
        self.opt_op = self.optimizer.minimize(self.loss, var_list=self.vars_list)
        self.opt_opb = self.optimizerb.minimize(self.lossb, var_list=self.weight)

        #self.opt_opb1 = self.optimizerb.minimize(self.lossb1, var_list=self.weight)
        self.opt_opc = self.optimizerc.minimize(self.lossc)
    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, weight, train_size,label_size, lambda1, lambda2, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.weight = weight
        self.train_size = train_size
        self.label_size = label_size
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.optimizerb = tf.train.AdamOptimizer(learning_rate=0.05)
        self.optimizerc = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer_bias = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()
    def _lossb(self, embedding, alpha, weight, mask, train_size, num):
        num = num
        loss = 0
        mask = tf.reshape(tf.cast(mask, dtype=tf.bool), [-1])
        embedding = tf.squeeze(embedding)
        embedding = tf.boolean_mask(embedding, mask, axis=0)

        alpha = tf.reshape(alpha, [1,-1])
        for p in range(num):
            cfeaturer = embedding
            if p == 0:
                cfeaturer = tf.concat((tf.zeros((train_size,1)), cfeaturer[:,1:num]), axis=1)
            elif p == num-1:                
                cfeaturer = tf.concat((cfeaturer[:,0:num-1], tf.zeros((train_size,1))), axis=1)
            else:                
                cfeaturer = tf.concat((cfeaturer[:,0:p], tf.zeros((train_size, 1)), cfeaturer[:, p+1:num]), axis=1)
            #tmp_a = tf.matmul(tf.matmul(tf.transpose(embedding[:, p:p+1]), tf.matrix_diag(weight)), cfeaturer)/train_size
            #tmp_b = tf.squeeze(tf.matmul(tf.transpose(embedding[:, p:p+1]),tf.reshape(weight, [-1,1])/train_size))*tf.matmul(tf.transpose(cfeaturer), tf.reshape(weight, [-1,1]))/train_size
            self.corr_tmp = tf.transpose(tf.matmul(tf.matmul(tf.transpose(embedding[:, p:p+1]), tf.matrix_diag(weight)), cfeaturer)/train_size) - (tf.squeeze(tf.matmul(tf.transpose(embedding[:, p:p+1]),tf.reshape(weight, [-1,1])/train_size))*tf.matmul(tf.transpose(cfeaturer), tf.reshape(weight, [-1,1]))/train_size) 
            loss += tf.reduce_sum(tf.pow(tf.matmul(tf.pow(alpha, 1), tf.abs(self.corr_tmp)), 2))

            #loss += tf.reduce_sum(tf.pow(self.corr_tmp, 2))
#            loss += tf.reduce_sum(tf.pow(tf.matmul(alpha, (tf.transpose(tf.matmul(tf.matmul(tf.transpose(embedding[:, p:p+1]), tf.matrix_diag(weight)), cfeaturer)/train_size) - tf.squeeze(tf.matmul(tf.transpose(embedding[:, p:p+1]),tf.reshape(weight, [-1,1])/train_size))*tf.matmul(tf.transpose(cfeaturer), tf.reshape(weight, [-1,1]))/train_size)), 2))
        loss += self.lambda1*tf.reduce_mean(tf.pow(weight, 2))
        loss += self.lambda2*tf.pow(tf.reduce_mean(weight)-1, 2)
        return loss
    def _lossb1(self, embedding, alpha, P, weight, mask, train_size, num):
        num = num
        loss = 0
        mask = tf.reshape(tf.cast(mask, dtype=tf.bool), [-1])
        embedding = tf.squeeze(embedding)
        embedding = tf.boolean_mask(embedding, mask, axis=0)
        alpha = tf.reshape(alpha, [1,-1])
        P = tf.reshape(P, [1, -1])

        alpha = alpha/tf.reduce_sum(alpha)
        new_alpha = tf.abs(alpha-P)


        for p in range(num):
            cfeaturer = embedding
            if p == 0:
                cfeaturer = tf.concat((tf.zeros((train_size,1)), cfeaturer[:,1:num]), axis=1)
            elif p == num-1:                
                cfeaturer = tf.concat((cfeaturer[:,0:num-1], tf.zeros((train_size,1))), axis=1)
            else:                
                cfeaturer = tf.concat((cfeaturer[:,0:p], tf.zeros((train_size, 1)), cfeaturer[:, p+1:num]), axis=1)
            #tmp_a = tf.matmul(tf.matmul(tf.transpose(embedding[:, p:p+1]), tf.matrix_diag(weight)), cfeaturer)/train_size
            #tmp_b = tf.squeeze(tf.matmul(tf.transpose(embedding[:, p:p+1]),tf.reshape(weight, [-1,1])/train_size))*tf.matmul(tf.transpose(cfeaturer), tf.reshape(weight, [-1,1]))/train_size
            self.corr_tmp = tf.transpose(tf.matmul(tf.matmul(tf.transpose(embedding[:, p:p+1]), tf.matrix_diag(weight)), cfeaturer)/train_size) - (tf.squeeze(tf.matmul(tf.transpose(embedding[:, p:p+1]),tf.reshape(weight, [-1,1])/train_size))*tf.matmul(tf.transpose(cfeaturer), tf.reshape(weight, [-1,1]))/train_size) 
            loss += tf.reduce_sum(tf.pow(tf.matmul(tf.pow(new_alpha, 1), tf.abs(self.corr_tmp)), 2))

            #loss += tf.reduce_sum(tf.pow(self.corr_tmp, 2))
#            loss += tf.reduce_sum(tf.pow(tf.matmul(alpha, (tf.transpose(tf.matmul(tf.matmul(tf.transpose(embedding[:, p:p+1]), tf.matrix_diag(weight)), cfeaturer)/train_size) - tf.squeeze(tf.matmul(tf.transpose(embedding[:, p:p+1]),tf.reshape(weight, [-1,1])/train_size))*tf.matmul(tf.transpose(cfeaturer), tf.reshape(weight, [-1,1]))/train_size)), 2))
        loss += self.lambda1*tf.reduce_mean(tf.pow(weight, 2))
        loss += self.lambda2*tf.pow(tf.reduce_mean(weight)-1, 2)
        return loss
    '''
    def _lossb(self, embedding, alpha, weight, mask, train_size, num):
        num = num
        loss = 0
        mask = tf.cast(mask, dtype=tf.bool)
        weight = tf.reshape(weight, [-1,1])
        embedding = tf.squeeze(embedding)
        embedding = tf.boolean_mask(embedding, mask, axis=0)
        cfeatureb = tf.sign(tf.sign(embedding)+1)
        mfeatureb = 1-cfeatureb
        tmp_list = []
        tmp_cfeaturer = []
        tmp_cfeatureb = []
        tmp_mfeatureb = []
        
        def f6():
            return tf.constant(0.0)
        for p in range(num):
            cfeaturer = embedding
            if p == 0:
                cfeaturer = cfeaturer[:,1:num]
            elif p == num-1:                
                cfeaturer = cfeaturer[:,0:num-1]
            else:                
                cfeaturer = tf.concat((cfeaturer[:,0:p], cfeaturer[:, p+1:num]), axis=1)
            #tmp_loss = tf.case({tf.not_equal(tf.squeeze(tf.matmul(tf.transpose(cfeatureb[:, p:p+1]), weight)), tf.constant(0.0)):f4, tf.not_equal(tf.squeeze(tf.matmul(tf.transpose(mfeatureb[:, p:p+1]), weight)), tf.constant(0.0)):f4}, default=f6)
            def f1():
                tmp_loss = tf.reduce_sum(tf.pow(tf.matmul(tf.transpose(cfeaturer), tf.multiply(mfeatureb[:, p:p+1], weight))/tf.matmul(tf.transpose(mfeatureb[:, p:p+1]), weight), 2))
                return tmp_loss
            def f2():
                tmp_loss = tf.reduce_sum(tf.pow(tf.matmul(tf.transpose(cfeaturer), tf.multiply(cfeatureb[:, p:p+1], weight))/tf.matmul(tf.transpose(cfeatureb[:, p:p+1]), weight), 2))
                return tmp_loss
            def f3():
                tmp_loss = tf.reduce_sum(tf.pow(tf.matmul(tf.transpose(cfeaturer), tf.multiply(cfeatureb[:, p:p+1], weight))/tf.matmul(tf.transpose(cfeatureb[:, p:p+1]), weight) - tf.matmul(tf.transpose(cfeaturer), tf.multiply(mfeatureb[:, p:p+1], weight))/tf.matmul(tf.transpose(mfeatureb[:, p:p+1]), weight), 2))
                return tmp_loss
            def f4():
                tmp_loss = tf.case([(tf.equal(tf.squeeze(tf.matmul(tf.transpose(cfeatureb[:, p:p+1]), weight)), tf.constant(0.0)), f1), (tf.equal(tf.squeeze(tf.matmul(tf.transpose(mfeatureb[:, p:p+1]), weight)), tf.constant(0.0)),f2)], default= f3) 
                return tmp_loss
            def f5():
                tmp_loss = tf.case({tf.equal(tf.squeeze(tf.matmul(tf.transpose(cfeatureb[:, p:p+1]), weight)), tf.constant(0.0)): f1(cfeaturer, mfeatureb, weight, p), tf.equal(tf.squeeze(tf.matmul(tf.transpose(mfeatureb[:, p:p+1]), weight)), tf.constant(0.0)): f2}, default=f3, exclusive=True) 
                return tmp_loss
            #a1 = f1()
            tmp_loss = tf.case([(tf.not_equal(tf.squeeze(tf.matmul(tf.transpose(cfeatureb[:, p:p+1]), weight)), tf.constant(0.0)),f4), (tf.not_equal(tf.squeeze(tf.matmul(tf.transpose(mfeatureb[:, p:p+1]), weight)), tf.constant(0.0)), f4)], default=f6)
            loss += tmp_loss
           
        loss += 1*tf.reduce_mean(tf.pow(weight, 2))
        loss += 100*tf.pow(tf.reduce_mean(weight)-1, 2)
        
        return loss
    '''
    def _loss(self):
        # Weight decay loss
        self.bias_loss = 0
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
            
            self.lossc += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #self.bias_loss += FLAGS.weight_decay * tf.nn.l2_loss(self.alpha)
        #self.bias_loss += FLAGS.weight_decay * tf.reduce_sum(tf.abs(self.alpha))
        # Cross entropy error
        #self.bias_loss = masked_sigmoid_cross_entropy(self.bias_logits, self.placeholders['bias_label'], self.placeholders['bias_mask'])
        self.loss += masked_softmax_cross_entropy_weight(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'], self.weight, self.label_size)
        #self.loss += 0.001 * tf.negative(tf.reduce_sum(tf.pow(self.hidden_embedding, 2)))
        self.lossc += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])
        self.optimizerb = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        #self.lossb = self._lossb(self.hidden_embedding, tf.multiply(self.weight, self.weight), self.placeholders['labels_mask'], self.train_size, self.hidden_embedding.shape.as_list()[1])

        self.lossb = self._lossb(self.hidden_embedding, self.trans_weight, tf.multiply(self.weight, self.weight), self.placeholders['labels_mask'], self.train_size, self.hidden_embedding.shape.as_list()[1])
        #self.lossb += 0. * masked_softmax_cross_entropy_weight(self.outputs, self.placeholders['labels'],
        #                                          self.placeholders['labels_mask'], self.weight, self.label_size)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        #self.bias_accuracy = bias_masked_accuracy(self.bias_logits_acc, self.placeholders['bias_label'], self.placeholders['bias_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))
        #self.layers.append(GraphConvolution(input_dim=16,
        #                                    output_dim=self.output_dim,
        #                                    placeholders=self.placeholders,
        #                                    act=lambda x: x,
        #                                    dropout=True,
        #                                    logging=self.logging))
        #self.layers.append(Dense(input_dim=16,
        #                         output_dim=self.output_dim,
        #                         placeholders=self.placeholders,
        #                         act=lambda x: x,
        #                         dropout=True,
        #                         logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
