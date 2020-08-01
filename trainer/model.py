"""Defines a Keras model and input function for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score, multilabel_confusion_matrix
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
import imageio

class DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator loading files from disk for model training.
    """
    def __init__(self, list_IDs, labels, image_dir, batch_size=32, dim=(224, 224), n_channels=3,
                 n_classes=1048, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.image_dir = image_dir

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __augment_img(self, img):
        'Scale, pad, and augment the image.'
        w = img.shape[0]
        h = img.shape[1]
        scale = min(self.dim[0] / w, self.dim[1] / h) # choose the largest scale down factor
        
        seq = iaa.Sequential([
            iaa.Resize(float(scale)),
            iaa.Cutout(),
            iaa.CenterPadToFixedSize(width=self.dim[0], height=self.dim[1]),
        ])
        img = seq(image=img) / 255 # reduce pixel values to [0,1]
        return img

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes)) # TODO: make dtype=object for 3 labels, but might lose their types
    
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Process and store image
            path = self.image_dir + '/'.join(ID[i] for i in range(4)) + '/' + ID
            img = imageio.imread(path)
            X[i,] = self.__augment_img(img) # TODO: does 512 improve performance? 

            # Store label
            y[i] = self.labels[ID][1]
            # TODO: when returning more than a single class
            # potential ValueError: failed to convert np array 
            # to tensor (unsupported object type tuple)

        return X, y

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

def biggest_logit(y_true, y_pred):
    return tf.math.reduce_max(y_pred)

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
    metrics = [accuracy_ml, biggest_logit] # TODO: determine which single metric to optimize, f1, hamming, etc.

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # logit input is [0, 1]
        metrics=metrics)
    return model
