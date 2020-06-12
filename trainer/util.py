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

MAX_IMG_SIZE = 512

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
    """ Stores LMDB labels into a dictionary. Each value of the LMDB is a dict.
    LMDB value
    - ingrs: list of integer ingredients (integers, 0 padded)
    - imgs: list of dicts of img: <img_id>.jpg, url: <url>
    - classes: integer category
    - intrs: array of shape (N, 1024) where N is the number of instructions and 
    there are 1024 float values for each component

    Args: 
    - file path to .mdb and .lock files

    Returns:
    - dict of img id, outcomes of interest
    """
    
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

        # Print progress    
        ctr += 1
        if ctr % 10000 == 0:
          print("# images saved: ", ctr)
          print("Time taken (min): ", (time.time()-start_time) / 60)
        
    return data_dict

def get_label(file_path, data_dict):
    """Return the ingredients label using the image id extracted 
    from file path name
    
    Args:
    - file path to image in directory
    - TF Static Hash Table mapping <img_id>.jpeg to extracted labels

    Returns:
    - label(s) 
    """
    # list out path components
    parts = tf.strings.split(file_path, os.path.sep) 
    return data_dict.lookup(parts[-1]) # last component is the image id

def decode_img(img):
    """Return 3D image tensor cast to float and shrunk (if necc.)
    
    """
    img = tf.image.decode_jpeg(img, channels=3) # converts to 3D uint8 tensor
    img = tf.image.convert_image_dtype(img, tf.float32) # scales down pixels to [0,1] range

    # Shrink image keeping aspect ratio if a dimension exceeds MAX_IMG_SIZE
    img_w = tf.cast(tf.shape(img)[0], dtype=tf.float32)
    img_h = tf.cast(tf.shape(img)[1], dtype=tf.float32)
    if img_w >= img_h:
        if img_w > MAX_IMG_SIZE: # when width the largest dim and too large  
            shrink = MAX_IMG_SIZE / img_w
            new_w = MAX_IMG_SIZE
            new_h = tf.math.floor(img_h * shrink)
            img = tf.image.resize(img, (new_w, new_h))
    else:
        if img_h > MAX_IMG_SIZE: # when height the largest dim and too large
            shrink = MAX_IMG_SIZE / img_h
            new_w = tf.math.floor(img_w * shrink)
            new_h = MAX_IMG_SIZE
            img = tf.image.resize(img, (new_w, new_h))

    return img

def process_path(file_path, data_dict):
    """Return image and label as tensors

    """
    label = get_label(file_path, data_dict)
    img = tf.io.read_file(file_path) # load the raw data from the file as a string
    img = decode_img(img)
    return img, label

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

