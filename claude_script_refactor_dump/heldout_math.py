"""Pure pieces of the held-out simulated-core experiment."""
import numpy as np


def split_train_valid(array, valid_fraction=0.2):
    """Split a sample along its first axis, preserving the notebook's tail hold-out convention."""
    n_valid = int(len(array) * valid_fraction)
    if n_valid < 1:
        raise ValueError('valid_fraction leaves no validation samples')
    return array[:-n_valid], array[-n_valid:]
