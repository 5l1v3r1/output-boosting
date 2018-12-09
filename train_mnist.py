"""
Train an MNIST classifier.
"""

import numpy as np
import tensorflow as tf

from output_boosting import learn, model_accuracy


def main():
    (x_train, y_train), (x_test, y_test) = load_dataset()
    y_train_oh = one_hot_labels(y_train)
    y_test_oh = one_hot_labels(y_test)
    x_train = x_train.reshape([-1, 28 * 28])
    x_test = x_test.reshape([-1, 28 * 28])
    model = learn(x_train, y_train_oh, depth=10, lr=1e-3, epochs=20,
                  batch_size=1000, log_fn=print)
    print('train accuracy %f' % model_accuracy(model, x_train, y_train_oh))
    print('test accuracy %f' % model_accuracy(model, x_test, y_test_oh))


def one_hot_labels(labels):
    oh = np.zeros([labels.shape[0], 10], dtype=np.float32)
    oh[np.arange(labels.shape[0]), labels] = 1.0
    return oh


def load_dataset():
    return tf.keras.datasets.mnist.load_data()


if __name__ == '__main__':
    main()
