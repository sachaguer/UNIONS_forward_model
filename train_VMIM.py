import time
start_time = time.time()

import os
import sys

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import jax_dataloader as jdl
from jaxili.inference import NPE, NLE

from neural_compression import NeuralCompressor
from neural_compression import CompressionCNN

from data_preprocessing import get_datasets_from_Hf

import matplotlib.pyplot as plt
from getdist import plots, MCSamples
import pandas as pd
from jaxili.utils import create_data_loader
from memory_profiler import profile

print("Device used by jax:", jax.devices())

data_path = "/lustre/fswork/projects/rech/prk/ujm83hk/sbi_input/"
compressed_data_path = data_path + "/compressed_data_VMIM.parquet"

compression_dataset_path = data_path + "/compression_data_noisereals3.parquet"
inference_dataset_path = data_path + "/inference_data_noisereals3.parquet"

num_cosmo_params = 2

nside = 512
n_ref = 1

lr = 1e-6
min_delta=1e-4
patience=15
num_epochs=200
batch_size=32







model_hparams = {
    "output_size": num_cosmo_params,
    
    "n_ref": n_ref,
    "nside": nside,
}

print("Loading dataset")
dataset = get_datasets_from_Hf(compression_dataset_path, train_test_split=None, num_cosmo_params=num_cosmo_params)
print(dataset)
VMIM = NPE()
print("Model created")

print("Adding simulations")
VMIM = VMIM.append_simulations_huggingface(dataset, train_test_split=[0.7, 0.25, 0.05])
del dataset
CHECKPOINT_PATH = os.path.abspath("./checkpoints/VMIM/")
print(f'Starting the training ({time.asctime(time.localtime(time.time()))})')

metrics_VMIM, density_estimator = VMIM.train(
    learning_rate=lr,
    patience=patience,
    min_delta=min_delta,
    z_score_x=False,
    training_batch_size=batch_size,
    checkpoint_path=CHECKPOINT_PATH,
    num_epochs=num_epochs,
    embedding_net=CompressionCNN,
    embedding_hparams=model_hparams,
)

print(f"Training Finished ({time.asctime(time.localtime(time.time()))})")

compressor_VMIM = density_estimator.embedding_net
print("Loading inference data")
inference_dataset = get_datasets_from_Hf(inference_dataset_path, train_test_split=None, num_cosmo_params=num_cosmo_params)
print(inference_dataset)
print("Compressing inference data")
def compress(batch):
    batch["t"] = compressor_VMIM(batch["x"])
    return batch
inference_dataset = inference_dataset.map(compress, batched=True, batch_size=batch_size)
inference_dataset = inference_dataset.remove_columns(["x"])
print(f"Saving compressed data to {compressed_data_path}")
print(inference_dataset)
inference_dataset.to_parquet(compressed_data_path)
print("Compressed data saved")

print(f"Execution took {(time.time() - start_time)//60}min")
