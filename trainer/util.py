"""Helper functions for preprocessing."""

import numpy as np
import pandas as pd
import tensorflow as tf
import imageio
import imgaug as ia
import imgaug.augmenters as iaa

import os
import sys
import pathlib
import time
import lmdb
import pickle
from collections import defaultdict

### Process Raw Labels

def get_valid_ids(img_dir, raw_ds):
    """ 
    Returns all paths to images with valid labels within the image directory.
    """

    valid_ids = list(raw_ds)
    dir_ids = [p.as_posix().split('/')[-1] for p in img_dir.rglob('*.jpg')]
    print("# original images: ", len(dir_ids))
    dir_ids = [i for i in dir_ids if i in valid_ids]
    print("# valid labeled images: ", len(dir_ids))
    return dir_ids

def encode_class(class_ids, id2class):
    """
    One hot encode raw class labels.
    """

    nclasses = len(list(id2class))
    nsamples = len(class_ids)
    one_hot = np.zeros((nsamples, nclasses))
    
    for i, class_id in enumerate(class_ids):
        one_hot[i][class_id-1] = 1 # adjustment for 1-indexed ids
    return one_hot

def find_unique_ingr_ids(raw_ds_list, invalid_ids):
    """
    Return unique ingredient labels excluding invalid ones from all the samples.
    """

    # Aggregate unique ids
    unique_ids = set()
    for ds in raw_ds_list:
        ingr_ds = np.array(list(ds.values()))[:,0] # extracting ingr ids from dict of img id : raw labels
        for ids in ingr_ds:
            nids = np.where(ids == 1)[0][0]
            unique_ids.update(ids[:nids])
    
    # Discard invalid ids
    for ingr_id in invalid_ids:
        unique_ids.remove(ingr_id)
         
    # Convert set to np array
    unique_ids = np.array(list(unique_ids))
    return unique_ids

def build_ingr_id_dict(ingr_ids, id2rvocab):
    """
    Assign a new id to each unique ingredient. Returns a new id to ingredient dict
    and an old id (rid) to new id (iid) dict. 
    """

    id2ingr = defaultdict(lambda:-1) # this default value allows for invalid key checking
    rid2iid = defaultdict(lambda:-1)
    for iid, rid in enumerate(ingr_ids):
        ingr_word = id2rvocab[rid]
        id2ingr[iid] = ingr_word
        rid2iid[rid] = iid
    return id2ingr, rid2iid

def encode_ingrs(ingr_ids_list, rid2iid):
    """
    One hot encode array of raw ingredient labels.
    """

    ningrs = len(list(rid2iid))
    nsamples = len(ingr_ids_list)
    one_hot = np.zeros((nsamples, ningrs))
    
    for i, ingr_ids in enumerate(ingr_ids_list):
        for rid in ingr_ids:
            if rid == 1: # stop token
                break
            iid = rid2iid[rid]
            if iid != -1: # checks if id found in dict (and thus a valid ingr)
                one_hot[i][iid] = 1
    return one_hot

### Read LMDB into raw labels dataset

def read_lmdb(file_path):
    """ 
    Stores LMDB labels into a dictionary. Each value of the LMDB is a dict.
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
    
    lmdb_env = lmdb.open(file_path, max_readers=1, readonly=True, lock=False,
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
            # TODO: save each set of labels as dict {'ingrs':ingrs, ...}, 
            # and modify data pipeline to index by name (e.g. 'ingrs', not col idx '0')
            data_dict[img['id']] = [ingrs, category, num_intrs]

        # Print progress    
        ctr += 1
        if ctr % 10000 == 0:
            print("# images saved: ", ctr)
            print("Time taken (min): ", (time.time()-start_time) / 60)
        
    return data_dict

### Main function for processing and loading data

def load_data():
    """
    Processes and returns all splits of data from the prespecified LMDB and directories.

    TODO: incorp 2 other splits
    TODO: can memory handle this? try on colab. maybe import sooner into tf dataset.
    """

    # Specify directories
    DATA_DIR = pathlib.Path(sys.path[0]).parents[0] / "data"
    IMG_DIR = DATA_DIR / "local_subset" 
    ID_DS_TRAIN_PATH = DATA_DIR / "data_dict_val.pkl"
    CLASS_PATH = DATA_DIR / "classes1M.pkl"
    RVOCAB_PATH = DATA_DIR / "vocab.txt"

    # Read in dict of image id : raw labels
    if not ID_DS_TRAIN_PATH.exists():
        print("Error: this raw dataset does not exist. Try extracting from the LMDB.")
        return
    
    id2raw_labels_tr = pickle.load(open(ID_DS_TRAIN_PATH.as_posix(), 'rb'))
    # id2raw_labels_val = 
    # id2raw_labels_te =
    
    img_ids = get_valid_ids(IMG_DIR, id2raw_labels_tr)

    raw_labels = np.array([id2raw_labels_tr[i] for i in img_ids])
    ingr_ids, class_ids, nsteps = raw_labels[:,0], raw_labels[:,1], raw_labels[:,2]

    # Build class id dictionary
    with open(CLASS_PATH, 'rb') as f:
        imid2clid = pickle.load(f) # image_id to class_id dict
        id2class = pickle.load(f) # class_id to class_name dict
    
    # One hot encode class ids
    class_labels = encode_class(class_ids, id2class)
    
    # Build ingr id dictionary
    with open(RVOCAB_PATH) as f_vocab:
        id2rvocab = {i+2: w.rstrip() for i, w in enumerate(f_vocab)}
        id2rvocab[1] = '</i>'
    rvocab2id = {v:k for k,v in id2rvocab.items()}
    
    invalid_ingr_names = ['Ingredients', '1', '100', '2', '200', '23', '30', '300', \
                          '4', '450', '50', '500', '6', '600']
    invalid_ids = [rvocab2id[n] for n in invalid_ingr_names]
    id_ds_list = [id2raw_labels_tr] # TODO: add id_ds_tr, import in up top
    unique_ids = find_unique_ingr_ids(id_ds_list, invalid_ids)
    id2ingr, rid2iid = build_ingr_id_dict(unique_ids, id2rvocab)
    print("# unique ingredients:", len(list(id2ingr)))
    
    # One hot encode ingr ids
    ingr_labels = encode_ingrs(ingr_ids, rid2iid)
    
    # Create new labels dictionary
    id2labels_tr = dict(zip(img_ids, zip(ingr_labels, class_labels, nsteps)))
    # id2labels_val = 
    # id2labels_te = 

    # Create new image id dictionary
    partition = {
        'train': img_ids[:18], 
        'validation': img_ids[18:]
    }

    return partition, id2labels_tr, id2class




