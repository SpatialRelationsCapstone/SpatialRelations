"""Classifier network layer that predicts relation between two objects."""

import tensorflow as tf

import os


class PredicateClassifier(object):
    """Network that classifies the relations between detected objects."""

    def __init__(self,
                 n_predicates=70,
                 learning_rate=0.001,
                 roi_pooling_dim=[4, 4],
                 var_scope="PredicateClassifier",
                 save_path="output/pred_cls.ckpt"):
        """Set hyperparameters, build graph."""
        self.n_predicates = n_predicates
        self.learning_rate = learning_rate
        self.roi_pooling_dim = roi_pooling_dim

        tf.reset_default_graph()
        with tf.variable_scope(var_scope):
            self._build_forward()
            self._build_optimization()

        # initialize session, checkpoints, variables
        self.sess = tf.Session()
        self.save_path = save_path
        self.saver = tf.train.Saver()

        if os.path.isfile(self.save_path + ".index"):
            self.saver.restore(self.sess, self.save_path)
        else:
            self.sess.run(tf.global_variables_initializer())

    def save_model(self):
        """Write tensorflow ckpt."""
        self.saver.save(self.sess, self.save_path)

    def _build_forward(self):
        # inputs
        self.image_feature_map = tf.placeholder(
            tf.float32,
            shape=[1, None, None, 315],
            name="image_feature_map")

        self.subject_bbox = tf.placeholder(
            tf.float32,
            shape=[1, 4],
            name="subject_bbox")

        self.object_bbox = tf.placeholder(
            tf.float32,
            shape=[1, 4],
            name="object_bbox")

        # "union" of subject and object features (pool and concat)
        self.subject_feats = tf.image.crop_and_resize(
            image=self.image_feature_map,
            boxes=self.subject_bbox,
            box_ind=tf.zeros([1], tf.int32),
            crop_size=self.roi_pooling_dim,
            name="subject_feats")

        self.object_feats = tf.image.crop_and_resize(
            image=self.image_feature_map,
            boxes=self.object_bbox,
            box_ind=tf.zeros([1], tf.int32),
            crop_size=self.roi_pooling_dim,
            name="object_feats")

        self.relation_feats = tf.layers.flatten(
            tf.concat([self.subject_feats, self.object_feats],
                      axis=-1),
            name="relation_feats")

        self.fc1 = tf.layers.dense(
            self.relation_feats,
            units=1024,
            activation=tf.nn.relu,
            name="fc1")

        self.logits = tf.layers.dense(
            self.fc1,
            units=self.n_predicates,
            activation=None,
            name="logits")

        self.output = tf.nn.softmax(
            self.logits,
            name="output")

    def _build_optimization(self):
        self.target_index = tf.placeholder(
            tf.int32,
            shape=[1],
            name="target_index")

        self.target_labels = tf.one_hot(
            self.target_index,
            depth=self.n_predicates,
            name="target_labels")

        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=self.target_labels,
            logits=self.logits,
            name="loss")

        self.optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.loss)
