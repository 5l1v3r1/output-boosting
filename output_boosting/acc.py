"""
Accuracy estimation.
"""

import numpy as np


def model_accuracy(model, samples, labels):
    """
    Compute the fraction of correctly classified samples.
    """
    samples = model.add_features(samples)
    logits = model.logits(samples)
    preds = np.argmax(logits, axis=-1)
    actual = np.argmax(labels, axis=-1)
    return np.sum(preds == actual) / len(preds)
