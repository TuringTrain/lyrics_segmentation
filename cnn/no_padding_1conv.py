import tensorflow as tf

from cnn.nn import NN


class NoPadding1Conv(NN):
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

        # Reshape to use within a convolutional neural net.
        #   contrary to mnist example, it just adds the last dimension whichs is the amount of channels in the image,
        #   in our case its only one, if we will add more features for each line – they will go there
        with tf.name_scope('reshape'):
            x_image = tf.expand_dims(self.g_in, -1)

        # First convolutional layer - 2d convolutions with windows always capturing the borders
        with tf.name_scope('conv1'):
            features_conv1 = 64
            W_conv1 = self.weight_variable([window_size+1, window_size+1, 1, features_conv1])
            b_conv1 = self.bias_variable([features_conv1])
            h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID')
            h_conv1 = tf.nn.relu(h_conv1 + b_conv1)

        # Pooling layer - downsamples by window_size.
        with tf.name_scope('pool1'):
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, window_size, window_size, 1],
                                     strides=[1, window_size, window_size, 1], padding='VALID')

        # Dropout - controls the complexity of the model, prevents co-adaptation of features
        with tf.name_scope('conv1-dropout'):
            h_pool1_drop = tf.nn.dropout(h_pool1, 1.0-(1.0-self.g_dprob)/2)

        # Second convolutional layer - performs horizontal convolutions
        with tf.name_scope('conv2'):
            features_conv2 = 128
            W_conv2 = self.weight_variable([1, window_size, features_conv1, features_conv2])
            b_conv2 = self.bias_variable([features_conv2])
            h_conv2 = tf.nn.conv2d(h_pool1_drop, W_conv2, strides=[1, 1, 1, 1], padding='VALID')
            h_conv2 = tf.nn.relu(h_conv2 + b_conv2)

        # Pooling layer - downsamples to a pixel.
        with tf.name_scope('pool2'):
            pool_size = int(ssm_size / window_size) - window_size
            h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, pool_size, 1],
                                     strides=[1, 1, pool_size, 1], padding='VALID')

        # We have to either fix the ssm_size or do an average here
        fc1_size = 128
        fc1_input_size = features_conv2
        with tf.name_scope('fc1'):
            W_fc1 = self.weight_variable([fc1_input_size, fc1_size])
            b_fc1 = self.bias_variable([fc1_size])

            h_pool2_flat = tf.reshape(h_pool2, [-1, fc1_input_size])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of features
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.g_dprob)

        # Map the features to 2 classes
        with tf.name_scope('fc2'):
            W_fc2 = self.weight_variable([fc1_size, 2])
            b_fc2 = self.bias_variable([2])

            self.g_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.g_labels, logits=self.g_out)
            self.g_loss = tf.reduce_mean(losses)

        # Evaluation
        with tf.name_scope('evaluation'):
            self.g_results = tf.argmax(self.g_out, axis=1, output_type=tf.int32)
