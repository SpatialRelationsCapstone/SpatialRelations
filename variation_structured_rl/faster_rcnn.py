"""Object detector used to propose object instances for deep VRL."""

import tensorflow as tf

from conv_nets import zeiler_fergus as zf


class RPN(object):
    """Region proposal network taking convolutional feature maps as input."""

    def __init__(self,
                 input_layer,
                 balancing_param=10,
                 batch_size=256,
                 feature_dim=256,
                 proposals_per_region=9,
                 learning_rate=0.001,
                 momentum=0.9,
                 var_scope="RPN"):
        """Set hyperparameters and build graph."""
        self.balancing_param = balancing_param
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.proposals_per_region = proposals_per_region
        self.learning_rate = learning_rate
        self.momentum = momentum

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

        self.feature_map = tf.nn.relu(
            self.mini_network,
            name="feature_map")

        self.map_shape = tf.shape(self.feature_map)

        # object vs background classifier
        self.classifier = tf.layers.conv2d(
            self.feature_map,
            filters=self.proposals_per_region * 2,
            kernel_size=[1, 1],
            strides=[1, 1],
            name="classifier")

        self.cls_reshaped = tf.reshape(
            self.classifier,
            shape=[self.map_shape[1], self.map_shape[2],
                   self.proposals_per_region, 2],
            name="cls_reshaped")

        self.cls_softmax = tf.nn.softmax(
            self.cls_reshaped,
            axis=-1,
            name="cls_softmax")

        # anchor-based region proposals
        self.region_regressor = tf.layers.conv2d(
            self.feature_map,
            filters=self.proposals_per_region * 4,
            kernel_size=[1, 1],
            strides=[1, 1],
            name="region_regressor")

        self.reg_reshaped = tf.reshape(
            self.region_regressor,
            shape=[self.map_shape[1], self.map_shape[2],
                   self.proposals_per_region, 4])

    def _build_optimization(self):
        self.loss_mask = tf.placeholder(
            tf.float32,
            shape=[None, None, self.proposals_per_region, 1])

        self.cls_target = tf.placeholder(
            tf.int32,
            shape=[None, None, self.proposals_per_region],
            name="cls_target")

        self.reg_target = tf.placeholder(
            tf.float32,
            shape=[None, None, self.proposals_per_region, 4],
            name="reg_target")

        # classification loss
        cls_loss = tf.losses.log_loss(
            labels=tf.one_hot(self.cls_target, 2),
            predictions=self.cls_softmax,
            reduction=tf.losses.Reduction.NONE)

        self.cls_loss = tf.reduce_sum(
            cls_loss * self.loss_mask,
            name="cls_loss")

        # regression loss (smooth L1 loss)
        reg_diff = tf.abs(self.reg_reshaped - self.reg_target)
        reg_loss = tf.keras.backend.switch(reg_diff < 0.5,
                                           0.5 * reg_diff ** 2,
                                           0.5 * (reg_diff - 0.5 * 0.5))
        pos_mask = tf.expand_dims(tf.cast(self.cls_target, tf.float32), -1)
        self.reg_loss = tf.reduce_sum(
            reg_loss * self.loss_mask * pos_mask,
            name="reg_loss")

        # final multitask loss
        n_anchors = tf.cast(tf.reduce_sum(self.map_shape), tf.float32)
        self.loss = tf.add(
            (1 / self.batch_size) * self.cls_loss,
            self.balancing_param * (1 / n_anchors) * self.reg_loss,
            name="loss")

        # optimizer
        self.optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.momentum).minimize(self.loss)


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
