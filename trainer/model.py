"""Defines a Keras model and input function for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score, multilabel_confusion_matrix


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

def eval_as_np(fn, y_true, y_pred):
    return tf.numpy_function(fn, [y_true, tf.round(y_pred)], tf.double)

def accuracy_ml(y_true, y_pred):
  return eval_as_np(accuracy_score, y_true, y_pred)

def f1_score_macro(y_true, y_pred):
  return f1_score(y_true, y_pred, average='macro')

def f1_ml(y_true, y_pred):
  return eval_as_np(f1_score_macro, y_true, y_pred)

def hamming_ml(y_true, y_pred):
  return eval_as_np(hamming_loss, y_true, y_pred)

def prec_ml(y_true, y_pred):
  return eval_as_np(precision_score, y_true, y_pred)

def recall_ml(y_true, y_pred):
  return eval_as_np(recall_score, y_true, y_pred)

def create_keras_model(input_shape, learning_rate):
    """
    Returns Keras model for single class prediction with MobileNetV2 architecture 
    """

    keras_mobilenet_v2 = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False)
    keras_mobilenet_v2.trainable = False

    model = tf.keras.Sequential([
        keras_mobilenet_v2,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1048, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
    metrics = [accuracy_ml] # TODO: determine which single metric to optimize, f1, hamming, etc.

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # logit input is [0, 1]
        metrics=metrics)
    return model
