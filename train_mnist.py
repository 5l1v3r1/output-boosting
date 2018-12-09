"""
Train an MNIST classifier.
"""

import numpy as np
import tensorflow as tf

from output_boosting import learn


def main():
    (x_train, y_train), (x_test, y_test) = load_dataset()
    y_train_oh = np.zeros([y_train.shape[0], 10], dtype=np.float32)
    y_train_oh[np.arange(y_train.shape[0]), y_train] = 1.0
    x_train = x_train.reshape([-1, 28 * 28])
    x_test = x_test.reshape([-1, 28 * 28])
    learn(x_train, y_train_oh, depth=100, lr=1e-3, epochs=2, batch_size=1000, log_fn=print)


def load_dataset():
    return tf.keras.datasets.mnist.load_data()


if __name__ == '__main__':
    main()
