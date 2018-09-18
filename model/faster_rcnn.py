"""Object detector used to propose object instances for deep VRL."""

import tensorflow as tf

from conv_nets import zf


class RPN(object):
    """Region proposal network taking convolutional feature maps as input."""

    def __init__(self,
                 input_layer,
                 feature_dim=256,
                 proposals_per_region=9,
                 var_scope="RPN"):
        """Set hyperparameters and build graph."""
        self.feature_dim = feature_dim
        self.proposals_per_region = proposals_per_region

        self.input_layer = input_layer
        with tf.variable_scope(var_scope):
            self._build_forward()
            self._build_optimization()

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
        self.region_regressor = tf.layers.conv2d(
            self.feature,
            filters=self.proposals_per_region * 4,
            kernel_size=[1, 1],
            strides=[1, 1],
            name="region_regressor")

        reg_shape = tf.shape(self.region_regressor)

        self.reg_reshaped = tf.reshape(
            self.region_regressor,
            shape=[reg_shape[1], reg_shape[2],     # don't need for batch dim
                   self.proposals_per_region, 4])  # since no more conv layers

        # object vs background classifier
        self.classifier = tf.layers.conv2d(
            self.feature,
            filters=self.proposals_per_region * 2,
            kernel_size=[1, 1],
            strides=[1, 1],
            name="classifier")

        cls_shape = tf.shape(self.classifier)

        self.cls_reshaped = tf.reshape(
            self.classifier,
            shape=[cls_shape[1], cls_shape[2],
                   self.proposals_per_region, 2],
            name="cls_reshaped")

        self.cls_softmax = tf.nn.softmax(
            self.cls_reshaped,
            axis=-1,
            name="cls_softmax")

        self.cls_output = tf.reshape(
            self.cls_softmax,
            shape=tf.shape(self.classifier),
            name="cls_output")

        # TODO: gather proposals with a high enough object score

        # TODO: non-maximum suppression

        self.final_proposals = []

    def _build_optimization(self):
        # TODO: multi-task loss similar to Fast-RCNN, anchor-indexed
        pass


class RCNNDetector(object):
    """Bounding box regressor and object classifier taking regions as input."""

    def __init__(self, n_categories, regions, var_scope="Detector"):
        """Set hyperparameters and build graph."""
        self.n_categories = n_categories

        self.regions = regions
        with tf.variable_scope(var_scope):
            self._build_forward()
            self._build_optimization()

    def _build_forward(self):
        self.roi_pooling = tf.placeholder(
            tf.float32, shape=[None, 2048])  # TODO

        self.fc1 = tf.layers.dense(
            self.roi_pooling,
            units=4096,
            activation=tf.nn.relu,
            name="fc1")

        self.fc2 = tf.layers.dense(  # also used as one of the inputs to VRL
            self.fc1,
            units=4096,
            activation=tf.nn.relu,
            name="fc2")

        # output layers
        self.category_classifier = tf.layers.dense(
            self.fc2,
            units=self.n_categories,
            activation=tf.nn.softmax,
            name="category_classifier")

        self.regressor = tf.layers.dense(
            self.fc2,
            units=self.n_categories * 4,
            activation=None,
            name="regressor")

    def _build_optimization(self):
        # TODO: multi-task loss - classification loss + bbox regression loss
        pass


class FasterRCNN(object):
    """End-to-end region proposal and object classification/localization."""

    def __init__(self,
                 n_categories,
                 conv_net=zf.ZFNet,
                 rpn=RPN,
                 detector=RCNNDetector):
        """Initialize instances of the constituent modules."""
        self.conv_net = conv_net()
        self.rpn = rpn(self.conv_net.conv_output)
        self.detector = detector(n_categories, self.rpn.final_proposals)

    def train(self):
        """4-step training algorithm to learn shared features."""
        # TODO: initialize CNN and RPN weights, train

        # TODO: initialize new CNN and Detector weights, train

        # TODO: copy trained detector weights and freeze them, fine-tune RPN

        # TODO: CNN weights remain frozen, fine-tune Detector
        pass
