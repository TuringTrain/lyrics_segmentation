import tensorflow as tf

from cnn.nn import NN


class Dense(NN):
    def __init__(self, window_size, ssm_size):
        super().__init__()

        self.g_dprob = None
        self.g_results = None

        self.window_size = window_size
        self.ssm_size = ssm_size

        self.define(window_size, ssm_size)

    def define(self, window_size, ssm_size):
        # Input of size:
        #   batch_size x window_size x max_ssm_size
        # Labels of size:
        #   batch_size
        # Note that we do not fix the first dimension to allow flexible batch_size for evaluation / leftover samples
        with tf.name_scope('input'):
            self.g_in = tf.placeholder(tf.float32, shape=[None, 2*window_size, ssm_size], name="input")
            self.g_labels = tf.placeholder(tf.int32, shape=[None], name="labels")

        self.g_dprob = tf.placeholder(tf.float32)
        layers = 4
        fc_input_size = 2*window_size * ssm_size
        fc_input = tf.reshape(self.g_in, [-1, fc_input_size])
        fc_size = 512
        for idx in range(layers):
            with tf.name_scope('fc%d' % idx):
                W_fc = self.weight_variable([fc_input_size, fc_size])
                b_fc = self.bias_variable([fc_size])

                fc_input = tf.nn.tanh(tf.matmul(fc_input, W_fc) + b_fc)

                # Dropout - controls the complexity of the model, prevents co-adaptation of features
                fc_input = tf.nn.dropout(fc_input, self.g_dprob)

            fc_input_size = fc_size
            fc_size = int(fc_size / 2)

        # Map the features to 2 classes
        with tf.name_scope('final'):
            W_fc2 = self.weight_variable([fc_input_size, 2])
            b_fc2 = self.bias_variable([2])

            self.g_out = tf.matmul(fc_input, W_fc2) + b_fc2

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.g_labels, logits=self.g_out)
            self.g_loss = tf.reduce_mean(losses)

        # Evaluation
        with tf.name_scope('evaluation'):
            self.g_results = tf.argmax(self.g_out, axis=1, output_type=tf.int32)
