"""Object detector used to propose object instances for deep VRL."""

import tensorflow as tf

from conv_nets import zf


class RPN(object):
    """Region proposal network taking convolutional feature maps as input."""

    def __init__(self,
                 input_layer,
                 var_scope,
                 feature_dim=256,
                 proposals_per_region=9):
        """Set hyperparameters and build graph."""
        self.feature_dim = feature_dim
        self.proposals_per_region = proposals_per_region

        self.input_layer = input_layer
        with tf.variable_scope(var_scope):
            self._build_forward()

    def _build_forward(self):
        self.mini_network = tf.layers.conv2d(
            self.input_layer,
            filters=self.feature_dim,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding="SAME",
            name="mini_network")

        self.feature = tf.nn.relu(
            self.mini_network,
            name="feature")

        # anchor-based region proposals
        self.regressor = tf.layers.conv2d(
            self.feature,
            filters=self.proposals_per_region * 4,
            kernel_size=[1, 1],
            strides=[1, 1],
            name="regressor")

        # object vs background classifier
        self.classifier = tf.layers.conv2d(
            self.feature,
            filters=self.proposals_per_region * 2,
            kernel_size=[1, 1],
            strides=[1, 1],
            name="classifier")


class RCNNDetector(object):
    """Bounding box regressor and object classifier taking regions as input."""

    def __init__(self, var_scope):
        """Set hyperparameters and build graph."""
        with tf.variable_scope(var_scope):
            self._build_forward()

    def _build_forward(self):
        raise NotImplementedError  # TODO


class FasterRCNN(object):
    """End-to-end region proposal and object classification/localization."""

    def __init__(self,
                 conv_net=zf.ZFNet,
                 rpn=RPN,
                 detector=RCNNDetector):
        """Initialize instances of the constituent modules."""
        self.conv_net = conv_net("ZFNet")
        self.rpn = rpn(self.conv_net.conv_output, "RPN")
        self.detector = detector("Detector")

    def train(self):
        """4-step training algorithm to learn shared features."""
        raise NotImplementedError  # TODO

    def non_max_suppression(self):
        """Reduce number of regions output by RPN to be fed into detector."""
        raise NotImplementedError  # TODO
