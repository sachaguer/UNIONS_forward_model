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
compressed_data_path = data_path + "/compressed_data_MSE.parquet"

compression_dataset_path = data_path + "/compression_data_noisereals3.parquet"
inference_dataset_path = data_path + "/inference_data_noisereals3.parquet"

num_cosmo_params = 2

nside = 512
n_ref = 1

lr = 1e-4
min_delta=1e-4
patience=15
num_epochs=60
batch_size=32







print("Creating dataset")
datasets = get_datasets_from_Hf(compression_dataset_path, [70, 25, 5], num_cosmo_params=num_cosmo_params)
print(datasets)
train_dataloader, valid_dataloader, test_dataloader = create_data_loader(
    datasets['train'], datasets['valid'], datasets['test'],
    train=[True, False, False],
    batch_size=batch_size,
)
del datasets
print("Dataloaders created")
def loss_mse(model, params, batch):
    xs, thetas = batch
    compressed_xs = model.apply({"params": params}, xs)
    return jnp.mean((compressed_xs - thetas)**2)

model_hparams = {
    "output_size": num_cosmo_params,
    
    "n_ref": n_ref,
    "nside": nside,
}

optimizer_hparams = {
    "lr": lr,
}

CHECKPOINT_PATH = os.path.abspath("./checkpoints")
logger_params = {
    "log_dir": os.path.join(CHECKPOINT_PATH, "Compressor/"),
}

exmp_input = (
    jnp.ones((11, 512*512*2)),
    jnp.zeros((11, 7)),
)

neural_compressor = NeuralCompressor(
    model_class=CompressionCNN,
    model_hparams=model_hparams,
    optimizer_hparams=optimizer_hparams,
    loss_fn=loss_mse,
    exmp_input=exmp_input,
    logger_params=logger_params,
    enable_progress_bar=True,
)

print("Model created")
print(f"Initial loss: {neural_compressor.eval_model(test_dataloader)['loss']}")
print(f'Starting the training ({time.asctime(time.localtime(time.time()))})')

metrics_MSE = neural_compressor.train_model(
    train_loader=train_dataloader,
    val_loader=valid_dataloader,
    num_epochs=num_epochs,
    min_delta=min_delta,
    patience=patience,
)

print(f"Training Finished ({time.asctime(time.localtime(time.time()))})")

print(f"Final loss: {neural_compressor.eval_model(test_dataloader)['loss']}")

compressor_MSE = neural_compressor.bind_model()
del test_dataloader
del train_dataloader
del valid_dataloader
print("Loading inference data")
inference_dataset = get_datasets_from_Hf(inference_dataset_path, train_test_split=None, num_cosmo_params=num_cosmo_params)
print(inference_dataset)
print("Compressing inference data")
def compress(batch):
    batch["t"] = compressor_MSE(batch["x"])
    return batch
inference_dataset = inference_dataset.map(compress, batched=True, batch_size=batch_size)
inference_dataset = inference_dataset.remove_columns(["x"])
print(f"Saving compressed data to {compressed_data_path}")
print(inference_dataset)
inference_dataset.to_parquet(compressed_data_path)
print("Compressed data saved")

print(f"Execution took {(time.time() - start_time)//60}min")