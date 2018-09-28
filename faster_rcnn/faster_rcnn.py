"""Object detector used to propose object instances for deep VRL."""

import tensorflow as tf

from conv_nets import zeiler_fergus as zf


class RPN(object):
    """Region proposal network taking convolutional feature maps as input."""

    def __init__(self,
                 image_input,
                 conv_output,
                 balancing_param=10,
                 batch_size=256,
                 feature_dim=256,
                 proposals_per_region=9,
                 nms_max_output_size=4000,
                 nms_threshold=0.7,
                 learning_rate=0.001,
                 momentum=0.9,
                 var_scope="RPN"):
        """Set hyperparameters and build graph."""
        # hyperparameters
        self.balancing_param = balancing_param
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.proposals_per_region = proposals_per_region
        self.nms_max_output_size = nms_max_output_size
        self.nms_threshold = nms_threshold
        self.learning_rate = learning_rate
        self.momentum = momentum

        # tensorflow operations
        self.image_input = image_input
        self.conv_output = conv_output
        with tf.variable_scope(var_scope):
            self._build_forward()
            self._build_optimization()
            self._build_nms_output()

    def train_step(self, sess, feed_dict):
        """Run a single iteration of gradient descent."""
        sess.run(self.optimizer.minimize(self.loss))

    def _build_forward(self):
        self.mini_network = tf.layers.conv2d(
            self.conv_output,
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

        self.reg_output = tf.reshape(
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
        reg_diff = tf.abs(self.reg_output - self.reg_target)
        reg_loss = tf.keras.backend.switch(reg_diff < 1,
                                           0.5 * reg_diff ** 2,
                                           reg_diff - 0.5)
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
            self.learning_rate, self.momentum)

    def _build_nms_output(self):
        # non max suppression
        self.anchors = tf.placeholder(
            tf.float32,
            shape=[None, None, self.proposals_per_region, 4])

        # tf.image.non_max_suppression requires coordinates (y1, x1, y2, x2)
        anchors_reshaped = tf.reshape(
            self.anchors,
            shape=[4, -1],
            name="anchors")

        proposals_reshaped = tf.reshape(
            self.reg_output,
            shape=[4, -1])

        y_a, x_a, h_a, w_a = tf.split(anchors_reshaped, 4)
        t_y, t_x, t_h, t_w = tf.split(proposals_reshaped, 4)

        y = t_y * h_a + y_a
        x = t_x * w_a + x_a
        h = h_a * tf.exp(t_h)
        w = w_a * tf.exp(t_w)

        # also normalize coordinates for ROI pooling operation later
        image_shape = tf.cast(tf.shape(self.image_input), tf.float32)

        y1 = (y - h // 2) / image_shape[1]
        x1 = (x - w // 2) / image_shape[2]
        y2 = (y + h // 2) / image_shape[1]
        x2 = (x + w // 2) / image_shape[2]

        self.proposal_coords = tf.reshape(
            tf.stack([y1, x1, y2, x2]),
            shape=[-1, 4],
            name="proposal_coords")

        # get scores from softmax layer
        softmax_reshaped = tf.reshape(
            self.cls_softmax,
            shape=[-1, 2])

        # softmax outputs are [P(not object), P(is object)]
        self.proposal_scores = softmax_reshaped[:, 1]

        nms_indices = tf.image.non_max_suppression(
            boxes=self.proposal_coords,
            scores=self.proposal_scores,
            max_output_size=self.nms_max_output_size,
            iou_threshold=self.nms_threshold)

        self.nms_proposals = tf.gather(
            self.proposal_coords,
            indices=nms_indices,
            name="nms_proposals")


class RCNNDetector(object):
    """Bounding box regressor and object classifier taking regions as input."""

    def __init__(self,
                 image_feats,
                 proposals,
                 n_categories,
                 balancing_param=1,
                 learning_rate=0.001,
                 momentum=0.9,
                 roi_pooling_dim=[4, 4],
                 var_scope="Detector"):
        """Set hyperparameters and build graph."""
        self.n_categories = n_categories

        # tensorflow operations
        self.image_feats = image_feats
        self.proposals = proposals

        # hyperparameters
        self.balancing_param = balancing_param
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.roi_pooling_dim = roi_pooling_dim

        with tf.variable_scope(var_scope):
            self._build_forward()
            self._build_optimization()

    def train_step(self, sess, feed_dict):
        """Run a single iteration of gradient descent."""
        sess.run(self.optimizer.minimize(self.loss))

    def _build_forward(self):
        # ROI pooling (implemented by crop_and_resize operation)
        # since ROIs are normalized, can be applied directly to feature map
        self.roi_pooling = tf.image.crop_and_resize(
            image=self.image_feats,
            boxes=self.proposals,
            box_ind=tf.zeros(tf.shape(self.proposals)[0], tf.int32),
            crop_size=self.roi_pooling_dim,
            name="roi_pooling")

        flat = tf.layers.flatten(self.roi_pooling)

        self.fc1 = tf.layers.dense(
            flat,
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
            units=self.n_categories + 1,
            activation=tf.nn.softmax,
            name="category_classifier")

        self.regressor = tf.layers.dense(
            self.fc2,
            units=self.n_categories * 4,
            activation=None,
            name="regressor")

        self.reg_output = tf.reshape(
            self.regressor,
            shape=[-1, self.n_categories, 4],
            name="reg_output")

    def _build_optimization(self):
        self.cls_target = tf.placeholder(
            tf.int32,
            shape=[None],
            name="cls_target")

        self.reg_target = tf.placeholder(
            tf.float32,
            shape=[None, self.n_categories, 4],
            name="reg_target")

        # classification loss
        cls_target_one_hot = tf.one_hot(
            self.cls_target,
            self.n_categories + 1)

        self.cls_pred = tf.reduce_sum(
            cls_target_one_hot * self.category_classifier,
            axis=-1,
            name="cls_pred")

        nonzero_preds = tf.gather(self.cls_pred, tf.where(self.cls_pred > 0))

        self.cls_loss = tf.reduce_mean(
            -tf.log(nonzero_preds),
            name="cls_loss")

        # regression loss
        reg_loss_mask = tf.expand_dims(
            tf.one_hot(self.cls_target - 1, self.n_categories),
            axis=-1)

        reg_diff = tf.abs(self.reg_output - self.reg_target)
        reg_loss = tf.keras.backend.switch(reg_diff < 1,
                                           0.5 * reg_diff ** 2,
                                           reg_diff - 0.5)

        self.reg_loss = tf.reduce_mean(
            reg_loss * reg_loss_mask,
            name="reg_loss")

        # multitask loss
        self.loss = tf.add(
            self.cls_loss,
            self.balancing_param * self.reg_loss,
            name="loss")

        # optimizer
        self.optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.momentum)


class FasterRCNN(object):
    """End-to-end region proposal and object classification/localization."""

    def __init__(self,
                 n_categories,
                 conv_net=zf.ZFNet,
                 rpn=RPN,
                 detector=RCNNDetector):
        """Initialize instances of the constituent modules."""
        self.conv_net = conv_net()

        self.rpn = rpn(self.conv_net.input,
                       self.conv_net.conv_output)

        self.detector = detector(self.rpn.feature_map,
                                 self.rpn.nms_proposals,
                                 n_categories)

    def train(self):
        """4-step training algorithm to learn shared features."""
        # TODO: initialize CNN and RPN weights, train

        # TODO: initialize new CNN and Detector weights, train

        # TODO: copy trained detector weights and freeze them, fine-tune RPN

        # TODO: CNN weights remain frozen, fine-tune Detector
        pass
