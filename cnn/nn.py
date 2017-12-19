import tensorflow as tf


class NN(object):
    def __init__(self):
        self.g_in = None
        self.g_labels = None
        self.g_out = None
        self.g_loss = None

    @staticmethod
    def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
