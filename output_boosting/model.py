"""
Output boosted models.
"""

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    @abstractmethod
    def add_features(self, samples):
        """
        Turn the samples into inputs for this model by
        adding the necessary features.

        Args:
          samples: a 2-D array where the outer dimension
            is the batch.

        Returns:
          A 2-D array where the inner dimension has been
            augmented.
        """
        pass

    @abstractmethod
    def logits(self, samples):
        """
        Compute the output logits for some samples.

        Args:
          samples: a 2-D array of augmented samples from
            add_features().

        Returns:
          A 2-D array of output logits.
        """
        pass

    def augmented_features(self, samples):
        """
        Get a set of features that includes information
        from the output of this Model.
        """
        samples = self.add_features(samples)
        logits = self.logits(samples)
        probs = softmax(logits)
        return np.concatenate([samples, probs], axis=-1)


class BaseModel(Model):
    """
    A straightforward linear Model with no augmented
    features.
    """

    def __init__(self, weights, biases):
        assert len(weights.shape) == 2
        assert len(biases.shape) == 1
        self.weights = weights
        self.biases = biases

    def add_features(self, samples):
        return samples

    def logits(self, samples):
        return (samples @ self.weights) + self.biases


class RecursiveModel(BaseModel):
    """
    A Model that bootstraps off a previous Model's
    predictions.
    """

    def __init__(self, parent, weights, biases):
        super().__init__(weights, biases)
        self.parent = parent

    def add_features(self, samples):
        return self.parent.augmented_features(samples)


def softmax(logits):
    maxes = np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(logits - maxes)
    sums = np.sum(exps, axis=-1, keepdims=True)
    return exps / sums
