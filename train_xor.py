"""
A demonstration on the xor problem.

This technique only works on xor because of randomness.
"""

import numpy as np

from output_boosting import learn


def main():
    learn(samples=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32),
          labels=np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32),
          depth=1000,
          lr=1e-4,
          batch_size=4,
          epochs=10,
          log_fn=print)


if __name__ == '__main__':
    main()
