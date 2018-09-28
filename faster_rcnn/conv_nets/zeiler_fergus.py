"""Zeiler-Fergus model with 5 shareable convolutional layers."""

import tensorflow as tf


class ZFNet(object):
    """8 layer convnet model that accepts variable input size."""

    def __init__(self, var_scope="ZFNet"):
        """Set hyperparameters and build graph."""
        with tf.variable_scope(var_scope):
            self._build_forward()

    def _build_forward(self):
        self.input = tf.placeholder(
            tf.float32,
            [None, None, None, 3],
            name="input")

        # layer 1
        self.conv1 = tf.layers.conv2d(
            self.input,
            filters=96,
            kernel_size=[7, 7],
            strides=[2, 2],
            padding="SAME",
            name="conv1")

        self.activation1 = tf.nn.relu(
            self.conv1,
            name="activation1")

        self.pool1 = tf.nn.max_pool(
            self.activation1,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool1")

        # layer 2
        self.conv2 = tf.layers.conv2d(
            self.pool1,
            filters=256,
            kernel_size=[5, 5],
            strides=[2, 2],
            padding="SAME",
            name="conv2")

        self.activation2 = tf.nn.relu(
            self.conv2,
            name="activation2")

        self.pool2 = tf.nn.max_pool(
            self.activation2,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding="SAME",
            name="pool2")

        # layer 3
        self.conv3 = tf.layers.conv2d(
            self.pool2,
            filters=384,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="SAME",
            name="conv3")

        self.activation3 = tf.nn.relu(
            self.conv3,
            name="activation3")

        # layer 4
        self.conv4 = tf.layers.conv2d(
            self.activation3,
            filters=384,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="SAME",
            name="conv4")

        self.activation4 = tf.nn.relu(
            self.conv4,
            name="activation4")

        # layer 5 (is taken as input by the RPN and Fast RCNN detector)
        self.conv5 = tf.layers.conv2d(
            self.activation4,
            filters=256,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="SAME",
            name="conv5")

        self.conv_output = tf.nn.relu(
            self.conv5,
            name="conv_output")
