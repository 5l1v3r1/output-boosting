"""
An ML algorithm that uses iterated linear classification.
"""

from .model import BaseModel, RecursiveModel
from .train import learn, learn_linear

__all__ = ['BaseModel', 'RecursiveModel', 'learn', 'learn_linear']
