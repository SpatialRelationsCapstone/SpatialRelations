"""Deep RL model that learns values for a variation-structured action space."""

import numpy as np
import tensorflow as tf

from collections import deque


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


class VRL(object):
    """Deep Q network that estimates attribute, predicate and object values."""

    def __init__(self,
                 n_attributes,
                 n_predicates,
                 n_object_categories,
                 image_feats,
                 subject_feats,
                 object_feats,
                 phrase_embedding,
                 learning_rate=0.001,
                 discount_factor=0.9,
                 batch_size=64):
        """Initialize hyperparameters and dataset-dependent parameters."""
        self.n_attributes = n_attributes
        self.n_predicates = n_predicates
        self.n_object_categories = n_object_categories

        # inputs on tensorflow graph
        self.image_feats = image_feats
        self.subject_feats = subject_feats
        self.object_feats = object_feats
        self.phrase_embedding = phrase_embedding

        # hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size

        # build graph
        self._build()

        # TODO: initialize replay memory buffer, scheduling related parameters

    def _build(self):
        """Construct tensorflow graph."""
        self.state_features = tf.concat(
            values=[self.image_feats,
                    self.subject_feats,
                    self.object_feats,
                    self.phrase_embedding],
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

        # full action space
        self.attribute_actions = tf.layers.dense(
            inputs=self.fusion2,
            units=self.n_attributes,
            activation=None,
            name="attribute_actions")

        self.predicate_actions = tf.layers.dense(
            inputs=self.fusion2,
            units=self.n_predicates,
            activation=None,
            name="predicate_actions")

        self.object_category_actions = tf.layers.dense(
            innputs=self.fusion2,
            units=self.n_predicates,
            activation=None,
            name="object_category_actions")

        # TODO: build optimization portion of graph
