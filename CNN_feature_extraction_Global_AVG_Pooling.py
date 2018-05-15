import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, Global AVG-pooling and softmax layer.
    """
    def __init__(
      self, x, y,  sequence_length , num_classes, vocab_size,
      embedding_size, filter_size, num_filters):

        # Placeholders for input, output and dropout
        self.input_x = x
        self.input_y = y
        self.dropout_keep_prob = tf.Variable(0.5, name="dropout_keep_prob")
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.layers.average_pooling1d(
                    h,
                    pool_size=[filter_size,embedding_size],
                    strides=[1,1],
                    padding='VALID',
                    data_format='channels_last',
                    name='Global_AVG_pooling'
                )

        # Combine all the pooled features
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters])
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)