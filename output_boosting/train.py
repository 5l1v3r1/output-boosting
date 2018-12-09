"""
Generating recursive models from training data.
"""

import tensorflow as tf


def learn_linear(samples, labels, epochs=10, lr=1e-3):
    """
    Learn a linear model.

    Args:
      samples: training data, in a 2-D array.
      labels: desired outputs, in a 2-D array of one-hot
        labels.
      epochs: number of training epochs.
      lr: adam stepsize.

    Returns:
      A tuple (weights, biases):
        weights: a [dim x num_labels] array.
        biases: a 1-D array of size num_labels.
    """
    with tf.Graph().as_default():
        samples_const = tf.constant(samples, dtype=tf.float32)
        labels_const = tf.constant(labels, dtype=tf.float32)
        weights = tf.get_variable('weights', shape=[samples.shape[-1], labels.shape[-1]])
        biases = tf.get_variable('biases', shape=[labels.shape[-1]])
        logits = tf.matmul(samples_const, weights) + biases
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_const)
        optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for _ in range(epochs):
                sess.run(optim)
            return sess.run((weights, biases))
