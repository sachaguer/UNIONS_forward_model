import numpy as np

import pandas as pd
import itertools
from datasets import load_dataset

def get_compressed_simulations_from_parquet(path):
    """Returns numpy arrays from parquet file"""
    data = pd.read_parquet(path)
    t = np.stack(data.t)
    theta = np.stack(data.theta)
    print(f"Loaded {t.shape[0]} simulations")
    return t, theta

def get_datasets_from_Hf(data_files, train_test_split=[0.7, 0.2, 0.1], num_cosmo_params=2):
    dataset = load_dataset("parquet", data_files=data_files, streaming=False, split='train').with_format("numpy")
    def reshape_map(example):
        x = np.array([example["kappa_E"], example["kappa_B"]])
        x = np.moveaxis(x, 0, -1)
        x = x.flatten()
        example["x"] = x
        example["theta"] = np.array([example["sigma_8"], example["Omega_m"], example['w'], example['Omega_b'], example['h'], example['n_s'], example['m_nu']])[:num_cosmo_params]
        return example
    dataset = dataset.map(reshape_map)
    dataset = dataset.remove_columns(["h", "Omega_m", "Omega_b", "sigma_8", "n_s", "w", "m_nu", "A_s", "kappa_E", "kappa_B", "shape_map"])
    if(train_test_split is None):
        return dataset
        
    train_test_split = np.array(train_test_split)
    train_fraction, val_fraction, test_fraction = train_test_split/train_test_split.sum()
    dataset = dataset.train_test_split(test_size=test_fraction)
    dataset_valtrain = dataset["train"].train_test_split(test_size=val_fraction/(train_fraction+val_fraction))
    dataset['train'] = dataset_valtrain['train']
    dataset['valid'] = dataset_valtrain['test']
    del dataset_valtrain
    return dataset
