import tensorflow as tf

from cnn.nn import NN


class NoPadding1Conv(NN):
    def __init__(self, window_size, ssm_size, added_features_size, channels):
        super().__init__()

        self.g_dprob = None
        self.g_results = None

        self.window_size = window_size
        self.ssm_size = ssm_size
        self.added_features_size = added_features_size
        self.channels = channels

        self.define(window_size, ssm_size, added_features_size, channels)

    def define(self, window_size, ssm_size, added_features_size, channels):
        # Input of size:
        #   batch_size x window_size x max_ssm_size
        # Labels of size:
        #   batch_size
        # Note that we do not fix the first dimension to allow flexible batch_size for evaluation / leftover samples
        with tf.name_scope('input'):
            self.g_in = tf.placeholder(tf.float32, shape=[None, 2*window_size, ssm_size, channels], name="input")
            self.g_labels = tf.placeholder(tf.int32, shape=[None], name="labels")
            self.g_dprob = tf.placeholder(tf.float32, name="dropout_prob")
            self.g_added_features = tf.placeholder(tf.float32, shape=[None, added_features_size], name="additional_features")

        # Reshape to use within a convolutional neural net.
        #   contrary to mnist example, it just adds the last dimension whichs is the amount of channels in the image,
        #   in our case its only one, if we will add more features for each line – they will go there
        with tf.name_scope('reshape'):
            # x_image = tf.expand_dims(self.g_in, -1)
            #   no reshaping necessary as incoming tensor has number of channels as lowest rank
            x_image = self.g_in

        # First convolutional layer - 2d convolutions with windows always capturing the borders
        with tf.name_scope('conv1'):
            features_conv1 = 128
            W_conv1 = self.weight_variable([window_size+1, window_size+1, channels, features_conv1])
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
        fc_input_size = features_conv2 + added_features_size
        fc_size = 512
        print('image_features.shape:', tf.reshape(h_pool2, [-1, features_conv2]).shape)
        print('added_features.shape:', self.g_added_features.shape)
        fc_input = tf.concat(
            (
            tf.reshape(h_pool2, [-1, features_conv2]),
            self.g_added_features
            ),
            axis = 1
        )
        print('combined.shape:', fc_input.shape)
        for fc_id in range(3):
            with tf.name_scope('fc-%d' % fc_id):
                W_fc = self.weight_variable([fc_input_size, fc_size])
                b_fc = self.bias_variable([fc_size])

                h_fc = tf.nn.tanh(tf.matmul(fc_input, W_fc) + b_fc)
                fc_input = tf.nn.dropout(h_fc, self.g_dprob)
                fc_input_size = fc_size

        # Map the features to 2 classes
        with tf.name_scope('fc-softmax'):
            W_fcs = self.weight_variable([fc_size, 2])
            b_fcs = self.bias_variable([2])

            self.g_out = tf.matmul(fc_input, W_fcs) + b_fcs


        # Regularization
        weights = tf.trainable_variables()
        l2_regularizer = tf.contrib.layers.l2_regularizer(
            scale=0.001, scope=None
        )
        l2_reg = tf.contrib.layers.apply_regularization(l2_regularizer, weights)

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.g_labels, logits=self.g_out)
            self.g_loss = tf.reduce_mean(losses) + l2_reg

        # Evaluation
        with tf.name_scope('evaluation'):
            self.g_results = tf.argmax(self.g_out, axis=1, output_type=tf.int32)
