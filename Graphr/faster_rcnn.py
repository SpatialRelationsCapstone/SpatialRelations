"""Object detector used to propose object instances for deep VRL."""

import tensorflow as tf


class VGG16(object):
    """Deep CNN using the VGG16 architecture."""

    def __init__(self, learning_rate):
        """Initialize hyperparameters, build network."""
        self.learning_rate = learning_rate

        self._build()

        # TODO: initialize with pretrained weights

    def _build(self):
        self.inputs = tf.placeholder(
            dtype=tf.float32,
            shape=[None, 224, 224, 3])

        self.conv_layers1 = self._vgg_module(
            inputs=self.inputs,
            filters=64,
            n_conv_layers=2,
            scope="conv1")

        self.conv_layers2 = self._vgg_module(
            inputs=self.conv_layers1,
            filters=128,
            n_conv_layers=2,
            scope="conv1")

        self.conv_layers3 = self._vgg_module(
            inputs=self.conv_layers2,
            filters=256,
            n_conv_layers=3,
            scope="conv1")

        self.conv_layers4 = self._vgg_module(
            inputs=self.conv_layers3,
            filters=512,
            n_conv_layers=3,
            scope="conv1")

        self.conv5_3 = self._vgg_module(
            inputs=self.conv_layers4,
            filters=512,
            n_conv_layers=3,
            scope="conv5",
            pool=False)

        self.pool5 = tf.nn.max_pool(
            value=self.conv5_3,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="SAME")

        # fully connected layers
        self.fc1 = tf.layers.dense(
            inputs=self.pool5,
            units=4096,
            activation=tf.nn.relu)

        self.fc2 = tf.layers.dense(
            inputs=self.fc1,
            units=4096,
            activation=tf.nn.relu)

        # TODO: optimization portion of graph
        # what does the VRL paper mean by fc6 layer? typo?

    def _vgg_module(self,
                    inputs,
                    filters,
                    n_conv_layers,
                    scope,
                    pool=True):
        """2-3 convolutional layers followed by max pooling."""
        layer = inputs

        with tf.variable_scope(scope):
            for _ in range(n_conv_layers):
                layer = tf.layers.conv2d(
                    inputs=layer,
                    filters=filters,
                    kernel_size=[3, 3],
                    strides=[1, 1],
                    padding="SAME")

            if pool:
                pool = tf.nn.max_pool(
                    value=layer,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding="SAME")

                return pool
            else:
                return layer


class RPN(object):
    """Region proposal network taking VGG16.conv5_3 as input."""

    def __init__(self):
        pass
