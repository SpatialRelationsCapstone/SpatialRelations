"""Deep RL model that learns values for a variation-structured action space."""

import numpy as np
import os
import tensorflow as tf

from keras_yolo3.yolo import YOLO

from collections import deque


YOLO_PATH = "keras_yolo3/model_data/trained_weights_final.h5"
ANCHORS_PATH = "keras_yolo3/model_data/tiny_yolo_anchors.txt"
CLASSES_PATH = "keras_yolo3/model_data/vrd_classes.txt"


class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False)

        return [self.buffer[i] for i in index]


class DQN(object):
    """Deep Q network that estimates predicate and object values."""

    def __init__(self,
                 n_predicates,
                 n_object_categories,
                 learning_rate=0.001,
                 feature_map_dim=315,
                 roi_pooling_dim=[4, 4],
                 save_path=None,
                 var_scope="DQN"):
        """Initialize hyperparameters and dataset-dependent parameters."""
        self.n_predicates = n_predicates
        self.n_object_categories = n_object_categories
        self.roi_pooling_dim = roi_pooling_dim

        # hyperparameters
        self.learning_rate = learning_rate

        # setup model saver
        self.save_path = save_path
        if self.save_path:
            self.saver = tf.train.Saver()

        # build graph
        with tf.variable_scope(var_scope):
            self._build_forward()
            self._build_optimization()

    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path)

    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path)

    def _build_forward(self):
        """Construct tensorflow graph for forward pass (outputs Q values)."""
        # inputs
        self.input_feature_map = tf.placeholder(
            tf.float32,
            shape=[1, None, None, 315],
            name="input_feature_map")

        self.subject_bbox = tf.placeholder(
            tf.float32,
            shape=[1, 4],
            name="subject_bbox")

        self.object_bbox = tf.placeholder(
            tf.float32,
            shape=[1, 4],
            name="object_bbox")

        self.image_feats = tf.layers.flatten(
            self.input_feature_map,
            name="image_feats")

        self.subject_feats = tf.layers.flatten(
            tf.image.crop_and_resize(
                image=self.input_feature_map,
                boxes=self.subject_bbox,
                box_ind=tf.constant([0]),
                crop_size=self.roi_pooling_dim),
            name="subject_feats")

        self.object_feats = tf.layers.flatten(
            tf.image.crop_and_resize(
                image=self.input_feature_map,
                boxes=self.object_bbox,
                box_ind=tf.constant([0]),
                crop_size=self.roi_pooling_dim),
            name="object_feats")

        # state representation layers
        self.state_features = tf.concat(
            values=[self.image_feats,
                    self.subject_feats,
                    self.object_feats],
            axis=-1)

        self.fusion1 = tf.layers.dense(
            inputs=self.state_features,
            units=4096,
            activation=tf.nn.relu,
            name="fusion1")

        self.fusion2 = tf.layers.dense(
            inputs=self.fusion1,
            units=2048,
            activation=tf.nn.relu,
            name="fusion2")

        # Q values for the 2 types of actions
        self.predicate_Q = tf.layers.dense(
            inputs=self.fusion2,
            units=self.n_predicates,
            activation=None,
            name="predicate_Q")

        self.category_Q = tf.layers.dense(
            innputs=self.fusion2,
            units=self.n_predicates,
            activation=None,
            name="category_Q")

    def _build_optimization(self):
        """Construct graph for network updates."""
        # predicate gradient
        self.predicate_return = tf.placeholder(
            tf.float32,
            shape=[None],
            name="predicate_return")

        self.predicate_action = tf.placeholder(
            tf.float32,
            shape=[None, self.n_predicates],
            name="predicate_action")

        self.predicate_value = tf.reduce_sum(
            self.predicate_action * self.predicate_Q,
            axis=1,
            name="predicate_value")

        self.predicate_grad_coef = tf.stop_gradient(
            self.predicate_return - self.predicate_value,
            name="predicate_grad_coef")

        self.predicate_gradient = -tf.reduce_mean(
            self.predicate_grad_coef * self.predicate_value,
            name="predicate_gradient")

        # category gradient
        self.category_return = tf.placeholder(
            tf.float32,
            shape=[None],
            name="category_return")

        self.category_action = tf.placeholder(
            tf.float32,
            shape=[None, self.n_object_categories],
            name="category_action")

        self.category_value = tf.reduce_sum(
            self.category_action * self.category_Q,
            axis=1,
            name="category_value")

        self.category_grad_coef = tf.stop_gradient(
            self.category_return - self.category_value,
            name="category_grad_coef")

        self.category_gradient = -tf.reduce_mean(
            self.category_grad_coef * self.category_value,
            name="category_gradient")

        # optimization ops
        self.multitask_loss = self.predicate_gradient + self.category_gradient

        self.optimizer = tf.train.RMSPropOptimizer(
            self.learning_rate).minimize(self.multitask_loss)


class VRL(object):
    """Variation-structured RL model that builds directed semantic graphs."""

    def __init__(self,
                 n_predicates,
                 n_object_categories,
                 yolo_model=YOLO_PATH,
                 yolo_anchors=ANCHORS_PATH,
                 yolo_classes=CLASSES_PATH,
                 target_update_frequency=10000,
                 discount_factor=0.9,
                 learning_rate=0.001,
                 max_memory=2048,
                 batch_size=64,
                 save_path="./checkpoints/VRL.ckpt"):
        """Build online and target networks, initialize replay memory."""
        # RL hyperparameters
        self.target_update_frequency = target_update_frequency
        self.discount_factor = discount_factor

        # initialize object detector
        self.detector = YOLO(model_path=yolo_model,
                             anchors_path=yolo_anchors,
                             classes_path=yolo_classes)

        # build network
        network_args = [n_predicates, n_object_categories,
                        learning_rate]

        tf.reset_default_graph()
        self.DQN = DQN(*network_args, save_path=self.save_path)

        self.target_DQN = DQN(*network_args, var_scope="target_DQN")

        # initialize experience replay buffer
        self.memory = Memory(max_memory)
        self.batch_size = batch_size

        # initialize tensorflow session
        self.sess = tf.Session()
        if os.path.isfile(self.save_path + ".index"):
            self.DQN.load(self.sess)
        else:
            self.sess.run(tf.global_variables_initializer())

        self._update_target_network()

    def select_actions(self):
        """Select a triplet of actions using an epsilon-greedy strategy."""
        raise NotImplementedError  # TODO: implement

    def train_network(self):
        """Sample a minibatch, calculate returns, and train online network."""
        raise NotImplementedError  # TODO: implement

    def _update_target_network(self):
        online_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "DQN")
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "target_DQN")

        update_op = []
        for online_var, target_var in zip(online_vars, target_vars):
            update_op.append(target_var.assign(online_var))

        self.sess.run(update_op)
