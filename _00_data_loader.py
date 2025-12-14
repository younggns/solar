import os
import numpy as np
import pandas as pd
import random
import json
import pickle
from sklearn.model_selection import train_test_split

from _00_text_utils import remove_mode_text, convert_selftext, process_quoting

def dictlist2dict(dict_list):
    dl = {}
    for d in dict_list:
        for k, v in d.items():
            if k in dl:
                dl[k].append(v)
            else:
                dl[k] = [v]
    return dl

def get_data_split_author(data_dir, author, n_train, n_val, n_test, fold=0, return_dict=True):
    
    train_data_path = f"{os.getcwd()}/{data_dir}/{author}_train_{fold}.npy"
    val_data_path = f"{os.getcwd()}/{data_dir}/{author}_valid_{fold}.npy"
    test_data_path = f"{os.getcwd()}/{data_dir}/{author}_test.npy"

    # training data
    if os.path.exists(train_data_path):
        train_data = np.load(train_data_path, allow_pickle=True)
    else:
        raise NotImplementedError

    if os.path.exists(test_data_path):
        test_data = np.load(test_data_path, allow_pickle=True)
    else:
        raise NotImplementedError

    if os.path.exists(val_data_path):
        val_data = np.load(val_data_path, allow_pickle=True)
    else:
        raise NotImplementedError

    # select n samples
    train_data = train_data[:min(n_train, len(train_data))]
    val_data = val_data[:min(n_val, len(val_data))]
    test_data = test_data[:min(n_test, len(test_data))]

    if return_dict:
        return dictlist2dict(train_data), dictlist2dict(val_data), dictlist2dict(test_data)
    else:
        return train_data, val_data, test_data

if __name__ == "__main__":
    pass
