
from jaxili.inference import NPE
from jaxili.train import TrainerModule, TrainState
from jaxili.model import NDENetwork
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import jax.random as jr
import jax_dataloader as jdl
from tqdm.contrib.itertools import product

from typing import Any, Callable, Dict, Iterable, Optional, Union
import nvsmi
import time
from memory_profiler import profile


class NeuralCompressor(TrainerModule):
    def init_apply_fn(self):
        self.apply_fn = self.model
    def run_model_init(self, exmp_input: Any, init_rng: Any) -> Dict:
        return self.model.init(init_rng, exmp_input[0])
    def handle_hf_dataset(self, batch):
        return batch["x"], batch["theta"]
    @classmethod
    def load_from_checkpoints(
        cls, model_class: NDENetwork, checkpoint: str, exmp_input: Any, loss_function: Callable
    ) -> Any:
        import os
        import json
        
        hparams_file = os.path.join(checkpoint, "hparams.json")
        assert os.path.isfile(hparams_file), "Could not find hparams file."
        with open(hparams_file, "r") as f:
            hparams = json.load(f)
        assert (
            hparams["model_class"] == model_class.__name__
        ), "Model class does not match. Check that you are using the correct architecture."
        hparams.pop("model_class")
        hparams.pop("loss_fn")
        if "activation" in hparams["model_hparams"].keys():
            hparams["model_hparams"]["activation"] = jax_nn_dict[
                hparams["model_hparams"]["activation"]
            ]
        if "nde" in hparams["model_hparams"].keys():
            hparams["model_hparams"]["nde"] = jaxili_nn_dict[
                hparams["model_hparams"]["nde"]
            ]
        if not hparams["logger_params"]:
            hparams["logger_params"] = dict()
        hparams["logger_params"]["log_dir"] = checkpoint
        trainer = cls(model_class=model_class, exmp_input=exmp_input, loss_fn=loss_function, **hparams)
        trainer.load_model()
        return trainer
    def print_tabulate(self, exmp_input: Any):
        x = exmp_input[0]
        try:
            print(self.model.tabulate(jax.random.PRNGKey(0), x))
        except Exception as e:
            print(f"Could not tabulate model: {e}")
        
class CompressionCNN(nn.Module):
    output_size : int
    
    nside : int
    n_ref : int

    @nn.compact
    def __call__(self, x):
        
        x = x.reshape((-1, self.nside, self.nside, 2))

        x = nn.Conv(features=16, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3), padding='VALID')
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3), padding='VALID')
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(5, 5), padding='SAME')(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3), padding='VALID')
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        
        x = nn.Dense(features=self.output_size)(x)
        x = x.reshape((x.shape[0], -1))
        return x

class CompressionPowerSpectrum(nn.Module):
    num_hiddens : Iterable[int]
    output_size : int

    @nn.compact
    def __call__(self, x):
        
        conv = nn.Conv(features=4, kernel_size=(5,), padding='VALID')(x)
        conv = nn.avg_pool(conv, window_shape=(1, 2), strides=(1, 2), padding='VALID')
        conv = nn.relu(conv)
        x = jnp.concatenate([x, conv], axis=1)
        for num_hidden in self.num_hiddens:
            x = nn.Dense(features=num_hidden)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.output_size)(x)
        return x

class CompressionJoint(nn.Module):
    CNN_num_hiddens : Iterable[int]
    Cl_num_hiddens : Iterable[int]
    output_size : int

    num_images : int
    nside : int
    n_ref : int
    
    @nn.compact
    def __call__(self, x):
        CNN_input_size = self.nside//self.n_ref * self.nside//self.n_ref * self.num_images
        CNN_input = x[:, :CNN_input_size]
        Cl_input = x[:, CNN_input_size:]

        CNN_output = CompressionCNN(self.CNN_num_hiddens, self.output_size, self.num_images, self.nside, self.n_ref)(CNN_input)
        Cl_output = CompressionPowerSpectrum(self.Cl_num_hiddens, self.output_size)(Cl_input)

        x = jnp.concatenate([CNN_output, Cl_output], axis=1)
        # x = nn.Dense(features=self.output_size)(x)
        return x

