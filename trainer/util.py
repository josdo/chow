"""Utilities to download and preprocess the data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

import time
import lmdb

# Storage directory
DATA_DIR = os.path.join(tempfile.gettempdir(), 'local_subset')

# Download options.
DATA_URL = (
    'https://storage.googleapis.com/cloud-samples-data/ai-platform/census'
    '/data')
TRAINING_FILE = 'adult.data.csv'
EVAL_FILE = 'adult.test.csv'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

IMG_SIZE = 32   # All images will be resized to 160x160

def _download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format.

    The CSVs may use spaces after the comma delimters (non-standard) or include
    rows which do not represent well-formed examples. This function strips out
    some of these problems.

    Args:
      filename: filename to save url to
      url: URL of resource to download
    """
    temp_file, _ = urllib.request.urlretrieve(url)
    with tf.io.gfile.GFile(temp_file, 'r') as temp_file_object:
        with tf.io.gfile.GFile(filename, 'w') as file_object:
            for line in temp_file_object:
                line = line.strip()
                line = line.replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                line += '\n'
                file_object.write(line)
    tf.io.gfile.remove(temp_file)


def download(data_dir):
    """Downloads census data if it is not already present.

    Args:
      data_dir: directory where we will access/save the census data
    """
    tf.io.gfile.makedirs(data_dir)

    training_file_path = os.path.join(data_dir, TRAINING_FILE)
    if not tf.io.gfile.exists(training_file_path):
        _download_and_clean_file(training_file_path, TRAINING_URL)

    eval_file_path = os.path.join(data_dir, EVAL_FILE)
    if not tf.io.gfile.exists(eval_file_path):
        _download_and_clean_file(eval_file_path, EVAL_URL)

    return training_file_path, eval_file_path

def extract_LMDB(filepath):
    """ Stores LMDB labels into a dictionary
    Args: 
    - file path to .mdb and .lock files

    Returns:
    - dict of img id, outcomes of interest
    """
    # LMDB data specs
    # print(sample.keys()) # the keys are 'ingrs', 'imgs', 'classes', 'intrs'
    # print(sample['ingrs'])
    # print(sample['imgs']) # list of dictionaries where one dictionary corresponds to one picture (url -> url and id -> id)
    # print(sample['classes']) # this is one integer value
    # print(sample['intrs'].shape) # shape is going to be (N, 1024) where N is the number of instructions and there are 1024 float values for each component

    # cur_path = sys.path[-1]
    # # lmdb_path = 'datasets/val_lmdb/'
    # lmdb_path = 'datasets/test_lmdb/'

    lmdb_env = lmdb.open(filepath, max_readers=1, readonly=True, lock=False,
                                 readahead=False, meminit=False)
    lmdb_txn = lmdb_env.begin(write=False)
    lmdb_cursor = lmdb_txn.cursor()

    data_dict = {}
    ctr = 0

    start_time = time.time()
    for key, value in lmdb_cursor:
        sample = pickle.loads(value,encoding='latin1')
        
        # Extract labels
        num_intrs = sample['intrs'].shape[0]
        category = sample['classes']
        ingrs = sample['ingrs']

        # Store labels by img id
        for img in sample['imgs']:
          data_dict[img['id']] = [ingrs, category, num_intrs]
        print(data_dict)
        break

        # Print progress    
        ctr += 1
        if ctr % 10000 == 0:
          print("# images saved: ", ctr)
          print("Time taken (min): ", (time.time()-start_time) / 60)
        
    print(len(list(data_dict)))
    return data_dict

def get_label(file_path, data_dict):
    """Return the ingredients label using the image id extracted from file path name"""

    parts = tf.strings.split(file_path, os.path.sep) # list of path components # os.path.sep
    print(os.path.sep)
    print(parts)
    # parts = parts.numpy()
    # index = file_path.rsplit('/', 1)[-1] # parts[-1].decode('ascii')
    return data_dict[parts[-1]] # the last path component is the image id (and dict key)

def decode_img(img):
    """Return 3D image tensor using dimensions from get_image_size"""
    img = tf.image.decode_jpeg(img, channels=3) # converts to 3D uint8 tensor
    img = tf.image.convert_image_dtype(img, tf.float32) # converts pixels to [0,1] range
    # tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT]) # optional resize
    return img

def process_path(file_path, data_dict):
    """Return image and label as tensors"""
    print(file_path)
    # tf.print(file_path)
    label = get_label(file_path, data_dict)
    img = tf.io.read_file(file_path) # load the raw data from the file as a string
    # w, h = gis.get_image_size(file_path) # TODO: Make file_path into string
    img = decode_img(img)
    return img, label

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # label = tf.one_hot(label, depth=10) # TODO: make num_classes an ARG, one hot encode
    return image, label # tf.reshape(label, (-1, 1))


def standardize(dataframe):
    """Scales numerical columns using their means and standard deviation to get
    z-scores: the mean of each numerical column becomes 0, and the standard
    deviation becomes 1. This can help the model converge during training.

    Args:
      dataframe: Pandas dataframe

    Returns:
      Input dataframe with the numerical columns scaled to z-scores
    """
    dtypes = list(zip(dataframe.dtypes.index, map(str, dataframe.dtypes)))
    # Normalize numeric columns.
    for column, dtype in dtypes:
        if dtype == 'float32':
            dataframe[column] -= dataframe[column].mean()
            dataframe[column] /= dataframe[column].std()
    return dataframe


def load_data():
    """Loads data into preprocessed (train_x, train_y, eval_y, eval_y)
    dataframes.

    Returns:
      A tuple (train_x, train_y, eval_x, eval_y), where train_x and eval_x are
      Pandas dataframes with features for training and train_y and eval_y are
      numpy arrays with the corresponding labels.
    """

    # Download CIFAR 10 dataset
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
    # dataset_tr = tfds.load('cifar10', as_supervised=True, split='train') 
    # dataset_te = tfds.load('cifar10', as_supervised=True, split='test') 

    # Preprocess dataset
    train_y, test_y = to_categorical(train_y, 10), to_categorical(test_y, 10)

    dataset_tr = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    dataset_te = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    dataset_tr = dataset_tr.map(preprocess)
    dataset_te = dataset_te.map(preprocess)

    # train_x, train_y = dataset_tr
    # eval_x, eval_y = dataset_te

    return dataset_tr, dataset_te # train_x, train_y, eval_x, eval_y

