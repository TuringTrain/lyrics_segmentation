import tensorflow as tf

from cnn.nn import NN


class MnistLike(NN):
    def __init__(self, window_size, ssm_size):
        super().__init__()

        self.g_dprob = None

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
            self.g_in = tf.placeholder(tf.float32, shape=[None, 2*window_size, ssm_size])
            self.g_labels = tf.placeholder(tf.int32, shape=[None])

        # Reshape to use within a convolutional neural net.
        #   contrary to mnist example, it just adds the last dimension whichs is the amount of channels in the image,
        #   in our case its only one, if we will add more features for each line – they will go there
        with tf.name_scope('reshape'):
            x_image = tf.expand_dims(self.g_in, -1)

        # First convolutional layer - maps input to 32 feature maps.
        with tf.name_scope('conv1'):
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = self.max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = self.max_pool_2x2(h_conv2)

        # We have to either fix the ssm_size or do an average here
        fc1_size = 512
        fc1_input_size = int(ssm_size / 4) * 64
        with tf.name_scope('fc1'):
            W_fc1 = self.weight_variable([fc1_input_size, fc1_size])
            b_fc1 = self.bias_variable([fc1_size])

            h_pool2_flat = tf.reshape(h_pool2, [-1, fc1_input_size])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of features
        with tf.name_scope('dropout'):
            self.g_dprob = tf.placeholder(tf.float32)
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

    @staticmethod
    def conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
