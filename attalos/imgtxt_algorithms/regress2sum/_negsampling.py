import numpy as np
import tensorflow as tf

class NegSamplingModel(object):
    """
    Create a tensorflow graph that does regression to a target using a negative sampling loss function
    """
    def __init__(self, input_size,
                    w2v,
                    learning_rate=1.001,
                    hidden_units=[200,200],
                    optim_words=True,
                    optim_unseen=False,
                    use_batch_norm=True):
        self.model_info = dict()

         # Placeholders for data
        self.model_info['input'] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)

        # Inputs to cost function
        if optim_words:
            self.model_info['w2v'] = tf.Variable(w2v)
            w2vgraph=self.model_info['w2v']
            print "Word vectors to be optimized on the CPU"
        else:
            w2vgraph=w2v
        if optim_unseen:
            self.model_info['pos_vecs'] = tf.placeholder(dtype=tf.float32)
            self.model_info['neg_vecs'] = tf.placeholder(dtype=tf.float32)
        else:
            self.model_info['pos_ids'] = tf.placeholder(dtype=tf.int32)
            self.model_info['neg_ids'] = tf.placeholder(dtype=tf.int32)  
            self.model_info['y_truth'] = tf.transpose(tf.nn.embedding_lookup(w2vgraph,
                                                                             self.model_info['pos_ids']),
                                                                             perm=[1,0,2])
            self.model_info['y_neg'] = tf.transpose(tf.nn.embedding_lookup(w2vgraph,
                                                                           self.model_info['neg_ids']),
                                                                           perm=[1,0,2])

        # Construct fully connected layers
        layers = []
        layer = self.model_info['input']
        for i, hidden_size in enumerate(hidden_units[:-1]):
            layer = tf.contrib.layers.relu(layer, hidden_size)
            layers.append(layer)
            if use_batch_norm:
                layer = tf.contrib.layers.batch_norm(layer)
                layers.append(layer)

        # Output layer should always be linear
        layer = tf.contrib.layers.linear(layer, w2v.shape[1])
        layers.append(layer)

        self.model_info['layers'] = layers
        self.model_info['prediction'] = layer

        def meanlogsig(pred, truth):
            reduction_indicies = 2
            return tf.reduce_mean( tf.log( tf.sigmoid( tf.reduce_sum(pred*truth, reduction_indices=reduction_indicies))))
        
        pos_loss = meanlogsig(self.model_info['prediction'], self.model_info['y_truth'])
        neg_loss = meanlogsig(-self.model_info['prediction'], self.model_info['y_neg'])
        loss = -(pos_loss + neg_loss)

        self.model_info['loss'] = loss
        self.model_info['optimizer'] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        self.model_info['init_op'] = tf.initialize_all_variables()
        self.model_info['saver'] = tf.train.Saver()

    def initialize_model(self, sess):
        sess.run(self.model_info['init_op'])

    def predict(self, sess, x):
        return sess.run(self.model_info['prediction'], feed_dict={self.model_info['input']: x})

    def fit(self, sess, x, y, **kwargs):
        _, loss = sess.run([self.model_info['optimizer'], self.model_info['loss']],
                           feed_dict={
                               self.model_info['input']: x,
                               self.model_info['pos_ids']: y,
                               self.model_info['neg_ids']: kwargs['neg_word_ids']
                           })
        return loss

    def save(self, sess, model_output_path):
        self.model_info['saver'].save(sess, model_output_path)

    def load(self, sess, model_input_path):
        self.model_info['saver'].restore(sess, model_input_path)
