"""
Generating recursive models from training data.
"""

import numpy as np
import tensorflow as tf

from .model import BaseModel, RecursiveModel


def learn(samples, labels, depth=10, log_fn=lambda x: None, **learn_kwargs):
    """
    Learn a BaseModel for the dataset.

    Args:
      samples: training data, in a 2-D array.
      labels: desired outputs, in a 2-D array of one-hot
        labels.
      depth: number of linear models to train.
      log_fn: a function called with a string on each
        iteration.
      learn_kwargs: kwargs for learn_linear.
    """
    assert depth >= 1
    weights, biases, loss = learn_linear(samples, labels, **learn_kwargs)
    log_fn('depth 0: loss=%f' % loss)
    model = BaseModel(weights, biases)
    aug_samples = model.augmented_features(samples)
    for i in range(depth - 1):
        zeros = np.zeros([labels.shape[-1]] * 2, dtype=np.float32)
        init_weights = np.concatenate([model.weights, zeros], axis=0)
        weights, biases, loss = learn_linear(aug_samples, labels, **learn_kwargs,
                                             init_biases=model.biases, init_weights=init_weights)
        log_fn('depth %d: loss=%f' % (i + 1, loss))
        model = RecursiveModel(model, weights, biases)
        aug_samples = model.augmented_features(samples)
    return model


def learn_linear(samples, labels, epochs=10, lr=1e-3, init_weights=None, init_biases=None):
    """
    Learn a linear model.

    Args:
      samples: training data, in a 2-D array.
      labels: desired outputs, in a 2-D array of one-hot
        labels.
      epochs: number of training epochs.
      lr: adam stepsize.

    Returns:
      A tuple (weights, biases, loss):
        weights: a [dim x num_labels] array.
        biases: a 1-D array of size num_labels.
        loss: the mean loss on the training set.
    """
    weight_shape = [samples.shape[-1], labels.shape[-1]]
    bias_shape = [weight_shape[-1]]
    if init_weights is None:
        init_weights = np.random.normal(size=weight_shape).astype(np.float32)
    if init_biases is None:
        init_biases = np.random.normal(size=bias_shape).astype(np.float32)
    with tf.Graph().as_default():
        samples_const = tf.constant(samples, dtype=tf.float32)
        labels_const = tf.constant(labels, dtype=tf.float32)
        weights = tf.get_variable('weights', initializer=tf.constant(init_weights))
        biases = tf.get_variable('biases', initializer=tf.constant(init_biases))
        logits = tf.matmul(samples_const, weights) + biases
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_const)
        loss = tf.reduce_mean(losses)
        optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(epochs):
                sess.run(optim)
            return sess.run((weights, biases, loss))
