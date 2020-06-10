"""Defines a Keras model and input function for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def input_fn(dataset, shuffle, num_epochs, batch_size):
    """Generates an input function to be used for model training.

    Args:
      features: numpy array of features used for training or inference
      labels: numpy array of labels for each example
      shuffle: boolean for whether to shuffle the data or not (set True for
        training, False for evaluation)
      num_epochs: number of epochs to provide the data for
      batch_size: batch size for training

    Returns:
      A tf.data.Dataset that can provide data to the Keras model for training or
        evaluation
    """
    # if labels is None:
    #     inputs = features
    # else:
    #     inputs = (features, labels)
    # dataset = tf.data.Dataset.from_tensor_slices(inputs)
    num_examples = dataset.reduce(0, lambda x, _: x + 1)
    # dataset = tf.cast(dataset, tf.int64)

    if shuffle:
        dataset = dataset.shuffle(buffer_size= batch_size*10) # TODO: fix num_examples error
        # ValueError: Tensor conversion requested dtype int64 for Tensor with dtype int32: <tf.Tensor: shape=(), dtype=int32, numpy=50000>

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def create_keras_model(input_shape, learning_rate):
    """Creates Keras Model for Multi-Class Classification.

    The single output node + Sigmoid activation makes this a Logistic
    Regression.

    Args:
      input_shape: How many features the input has
      learning_rate: Learning rate for training

    Returns:
      The compiled Keras model (still needs to be trained)
    """

    keras_mobilenet_v2 = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False)
    keras_mobilenet_v2.trainable = False

    model = tf.keras.Sequential([
        keras_mobilenet_v2,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Custom Optimizer:
    # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    return model
