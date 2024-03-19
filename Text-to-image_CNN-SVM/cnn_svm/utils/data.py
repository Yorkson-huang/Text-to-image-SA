"""Utility functions module"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def load_tfds(
    name: str = "mnist"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_dataset = tfds.load(name=name, split=tfds.Split.TRAIN, batch_size=-1)
    train_dataset = tfds.as_numpy(train_dataset)

    train_features = train_dataset["image"]
    train_labels = train_dataset["label"]

    train_features = train_features.astype("float32")
    train_features = train_features / 255.0

    test_dataset = tfds.load(name=name, split=tfds.Split.TEST, batch_size=-1)
    test_dataset = tfds.as_numpy(test_dataset)

    test_features = test_dataset["image"]
    test_labels = test_dataset["label"]

    test_features = test_features.astype("float32")
    test_features = test_features / 255.0

    return train_features, test_features, train_labels, test_labels


def create_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    as_supervised: bool = True,
) -> tf.data.Dataset:

    if as_supervised:
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((features, features))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(features.shape[1])
    return dataset
