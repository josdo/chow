"""Pipeline reading and preprocessing data from the original files."""

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

def get_valid_ids(img_dir, raw_ds, other_ds=[], subdirect=False):
    """ 
    Returns all paths to images with valid labels within the image directory.
    """
    # Find all labeled ids
    valid_ids = list(raw_ds)
    start_time = time.time()

    # Find all ids from the directory
    if subdirect:
        subdir = "/a/a/a"
        print("Collecting all image paths in subdir {} of {}".format(subdir, img_dir))
        dir_paths = pathlib.Path(img_dir+subdir).rglob("*.jpg")
    else:
        print("Collecting all image paths in", img_dir)
        dir_paths = pathlib.Path(img_dir).rglob("*.jpg")
    
    dir_ids = [p.as_posix().split('/')[-1] for p in dir_paths]
    print("Collecting completed in {:.2f} minutes".format((time.time()-start_time)/60))

    # Check if directory ids belong to current partition
    print("{} original images".format(len(dir_ids)))
    valid_dir_ids = [i for i in dir_ids if i in valid_ids]
    print("{} usable labeled images".format(len(valid_dir_ids)))

    # Check if directory ids belong to current partition (for debugging)
    if len(other_ds) > 0:
        for ds in other_ds:
            other_dir_ids = [i for i in dir_ids if i in list(ds)]
            print("{} images labeled for another partition".format(len(other_dir_ids)))

    return valid_dir_ids

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

def dd(): # named function required for the default dict to be pickled later
    return -1

def build_ingr_id_dict(ingr_ids, id2rvocab):
    """
    Assign a new id to each unique ingredient. Returns a new id to ingredient dict
    and an old id (rid) to new id (iid) dict. 
    """

    id2ingr = defaultdict(dd) # this default value allows for invalid key checking
    rid2iid = defaultdict(dd)
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

def read_lmdb(lmdb_path):
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
    
    lmdb_env = lmdb.open(lmdb_path, max_readers=1, readonly=True, lock=False,
                                 readahead=False, meminit=False)
    lmdb_txn = lmdb_env.begin(write=False)
    lmdb_cursor = lmdb_txn.cursor()

    data_dict = {}
    ctr = 0

    print("Reading LMDB at ", lmdb_path)
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

def load_data(dirs, files):
    """
    TODO: add test partition, details in code
    Processes and returns all splits of data from the prespecified data directories.
    """

    # Specify data directories
    img_dir_tr, img_dir_val = dirs['imgs_val'], dirs['imgs_test']
    lmdb_tr_out_path, lmdb_val_out_path = files['val_pkl'], files['test_pkl']
    class_dict_path = files['classes_pkl']
    recipe_vocab_path = files['rvocab_pkl']

    # Read in dict of raw labels
    if not (os.path.isfile(lmdb_tr_out_path) and os.path.isfile(lmdb_val_out_path)):
        print("Error: one of the LMDB output files does not exist. Try re-reading the LMDBs.")
        return
    
    id2raw_labels_tr = pickle.load(open(lmdb_tr_out_path, 'rb'))
    id2raw_labels_val = pickle.load(open(lmdb_val_out_path, 'rb'))
    
    # TODO: clear subdirect arg, other ds search
    img_ids_tr = get_valid_ids(img_dir_tr, id2raw_labels_tr, [id2raw_labels_val], subdirect=True)
    img_ids_val = get_valid_ids(img_dir_val, id2raw_labels_val, [id2raw_labels_tr], subdirect=True)

    raw_labels_tr = np.array([id2raw_labels_tr[i] for i in img_ids_tr])
    raw_labels_val = np.array([id2raw_labels_val[i] for i in img_ids_val])
    
    # Combine ids and labels from separate partitions
    img_ids = np.concatenate([img_ids_tr, img_ids_val])
    raw_labels = np.concatenate([raw_labels_tr, raw_labels_val])

    # TODO: ^^ add test partition + \/ add id2raw_labels_te to id_ds_list

    # Separate raw labels by type
    ingr_ids, class_ids, nsteps = raw_labels[:,0], raw_labels[:,1], raw_labels[:,2]

    # Build class id dictionary
    with open(class_dict_path, 'rb') as f:
        imid2clid = pickle.load(f) # image_id to class_id dict
        id2class = pickle.load(f) # class_id to class_name dict
    
    # One hot encode class ids
    class_labels = encode_class(class_ids, id2class)
    
    # Build ingr id dictionary
    with open(recipe_vocab_path) as f_vocab:
        id2rvocab = {i+2: w.rstrip() for i, w in enumerate(f_vocab)}
        id2rvocab[1] = '</i>'
    rvocab2id = {v:k for k,v in id2rvocab.items()}
    
    invalid_ingr_names = ['Ingredients', '1', '100', '2', '200', '23', '30', '300', \
                          '4', '450', '50', '500', '6', '600']
    invalid_ids = [rvocab2id[n] for n in invalid_ingr_names]
    id_ds_list = [id2raw_labels_tr, id2raw_labels_val] # TODO: add test ids dict
    unique_ids = find_unique_ingr_ids(id_ds_list, invalid_ids)
    id2ingr, rid2iid = build_ingr_id_dict(unique_ids, id2rvocab)
    print("# unique ingredients:", len(list(id2ingr)))
    
    # One hot encode ingr ids
    ingr_labels = encode_ingrs(ingr_ids, rid2iid)
    
    # Create new labels dictionary
    labels = dict(zip(img_ids, zip(ingr_labels, class_labels, nsteps)))
   
    # Create new image id dictionary
    partition = {
        'train': img_ids_tr, 
        'validation': img_ids_val
    }

    return partition, labels, id2ingr




