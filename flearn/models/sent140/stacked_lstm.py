import json
import numpy as np
import tensorflow as tf
from tqdm import trange

from tensorflow.contrib import rnn

from flearn.utils.model_utils import batch_data, batch_data_multiple_iters
from flearn.utils.language_utils import line_to_indices
from flearn.utils.tf_utils import graph_size, process_grad

with open('flearn/models/sent140/embs.json', 'r') as inf:
    embs = json.load(inf)
id2word = embs['vocab']
word2id = {v: k for k,v in enumerate(id2word)}
word_emb = np.array(embs['emba'])

def process_x(raw_x_batch, max_words=25):
    x_batch = [e[4] for e in raw_x_batch]
    x_batch = [line_to_indices(e, word2id, max_words) for e in x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [1 if e=='4' else 0 for e in raw_y_batch]
    y_batch = np.array(y_batch)

    return y_batch

class Model(object):

    def __init__(self, seq_len, num_classes, n_hidden, optimizer, seed):
        #params
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.emb_arr = word_emb

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(123+seed)
            self.features, self.labels, self.train_op, self.grads, self.eval_metric_ops, self.loss = self.create_model(optimizer)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)

        # find memory footprint and compute cost of the model
        self.size = graph_size(self.graph)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            metadata = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            self.flops = tf.profiler.profile(self.graph, run_meta=metadata, cmd='scope', options=opts).total_float_ops
    
    def create_model(self, optimizer):
        features = tf.placeholder(tf.int32, [None, self.seq_len], name='features')
        labels = tf.placeholder(tf.int64, [None,], name='labels')

        embs = tf.Variable(self.emb_arr, dtype=tf.float32, trainable=False)
        x = tf.nn.embedding_lookup(embs, features)
        
        stacked_lstm = rnn.MultiRNNCell(
            [rnn.BasicLSTMCell(self.n_hidden) for _ in range(2)])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        fc1 = tf.layers.dense(inputs=outputs[:,-1,:], units=30)
        pred = tf.squeeze(tf.layers.dense(inputs=fc1, units=1))
        
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=pred)
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        
        correct_pred = tf.equal(tf.to_int64(tf.greater(pred,0)), labels)
        eval_metric_ops = tf.count_nonzero(correct_pred)
        
        return features, labels, train_op, grads, eval_metric_ops, loss

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def get_gradients(self, data, model_len):
        
        grads = np.zeros(model_len)
        num_samples = len(data['y'])
        processed_samples = 0

        if num_samples < 50:
            input_data = process_x(data['x'])
            target_data = process_y(data['y'])
            with self.graph.as_default():
                model_grads = self.sess.run(self.grads, 
                    feed_dict={self.features: input_data, self.labels: target_data})
                grads = process_grad(model_grads)
            processed_samples = num_samples

        else:  # calculate the grads in a batch size of 50
            for i in range(min(int(num_samples / 50), 4)):
                input_data = process_x(data['x'][50*i:50*(i+1)])
                target_data = process_y(data['y'][50*i:50*(i+1)])
                with self.graph.as_default():
                    model_grads = self.sess.run(self.grads,
                    feed_dict={self.features: input_data, self.labels: target_data})

                flat_grad = process_grad(model_grads)
                grads = np.add(grads, flat_grad) # this is the average in this batch

            grads = grads * 1.0 / min(int(num_samples/50), 4)
            processed_samples = min(int(num_samples / 50), 4) * 50

        return processed_samples, grads
    
    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            comp: number of FLOPs computed while training given data
            update: list of np.ndarray weights, with each weight array
        corresponding to a variable in the resulting graph
        '''
        
        for _ in trange(num_epochs, desc='Epoch: ', leave=False):
            for X,y in batch_data(data, batch_size):
                input_data = process_x(X, self.seq_len)
                target_data = process_y(y)
                with self.graph.as_default():
                    self.sess.run(self.train_op,
                        feed_dict={self.features: input_data, self.labels: target_data})
        soln = self.get_params()
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp

    def solve_iters(self, data, num_iters=1, batch_size=32):
        '''Solves local optimization problem'''

        for X, y in batch_data_multiple_iters(data, batch_size, num_iters):
            input_data = process_x(X, self.seq_len)
            target_data = process_y(y)
            with self.graph.as_default():
                self.sess.run(self.train_op, feed_dict={self.features: input_data, self.labels: target_data})
        soln = self.get_params()
        comp = 0
        return soln, comp
    
    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        x_vecs = process_x(data['x'], self.seq_len)
        labels = process_y(data['y'])
        with self.graph.as_default():
            tot_correct, loss = self.sess.run([self.eval_metric_ops, self.loss],
                feed_dict={self.features: x_vecs, self.labels: labels})
        return tot_correct, loss
    
    def close(self):
        self.sess.close()
