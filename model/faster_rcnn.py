"""Object detector used to propose object instances for deep VRL."""

import tensorflow as tf


class RPN(object):
    """Region proposal network taking convolutional feature maps as input."""

    def __init__(self,
                 input_layer,
                 var_scope):
        """Set hyperparameters and build graph."""
        self.input_layer = input_layer
        with tf.variable_scope(var_scope):
            self._build_forward()

    def _build_forward(self):
        raise NotImplementedError  # TODO


class RCNNDetector(object):
    """Bounding box regressor and object classifier taking regions as input."""

    def __init__(self,
                 var_scope):
        """Set hyperparameters and build graph."""
        with tf.variable_scope(var_scope):
            self._build_forward()

    def _build_forward(self):
        raise NotImplementedError  # TODO


class FasterRCNN(object):
    """End-to-end region proposal and object classification/localization."""

    def __init__(self,
                 rpn,
                 detector):
        """Initialize instances of the constituent modules."""
        self.rpn = rpn
        self.detector = detector

    def train(self):
        """4-step training algorithm to learn shared features."""
        raise NotImplementedError  # TODO
