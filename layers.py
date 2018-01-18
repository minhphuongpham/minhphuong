import tensorflow as tf


def weight_bias_variable(in_channels, out_channels, filter_size, name):
    W_shape = (filter_size, filter_size, in_channels, out_channels)
    initial_W = tf.truncated_normal(shape=W_shape, stddev= .1)
    initial_b = tf.constant(.1, shape=(out_channels, ))

    return tf.Variable(initial_W, name = name+"_W"),\
            tf.Variable(initial_b, name = name+"_b")

class ConvLayer(object):
    def __init__(self, input_layer, in_channels, out_channels, filter_size=3, strides = 1, padding = 'SAME',
                 activation = tf.nn.relu, name = None):

        self.W, self.b = weight_bias_variable(in_channels, out_channels, filter_size, name)
        conv = tf.nn.conv2d(input_layer.output, self.W, strides=[1, strides, strides, 1], padding = padding)
        h = tf.nn.bias_add(conv, self.b)

        if activation is not None:
            self.output = activation(h)
        else:
            self.output = h

class MaxPoolLayer(object):
    def __init__(self, input_layer, ksize=2, strides=1, padding = 'SAME', name = None):
        self.output = tf.nn.max_pool(
                input_layer.output, ksize=[1, ksize, ksize, 1], strides= [1, strides, strides, 1], padding=padding, name = name)


class DenseLayer(object):
    def __init__(self, input_layer, n_units, activation = tf.nn.relu, name = None):

        input_tensor = input_layer.output
        _, _, filter_width, in_channels = [i.value for i in input_tensor.get_shape()]
        self.layer = ConvLayer(input_layer, in_channels = in_channels, out_channels= n_units,
                               filter_size= filter_width, padding = 'VALID', activation = activation,
                               name = name)

        self.W = self.layer.W
        self.b = self.layer.b
        self.output = self.layer.output

class InputLayer(object):
    def __init__(self, x):
        self.output = x

class DropoutLayer(object):
    def __init__(self, input_layer, dropout_rate):
        self.output = tf.nn.dropout(input_layer.output, keep_prob=1 - dropout_rate)

