from generate_image import get_cells_from_map
from jaxili.inference import NPE
from jaxili.train import TrainerModule
from jaxili.model import NDENetwork
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import jax.random as jr
import jax_dataloader as jdl
from tqdm.contrib.itertools import product


from typing import Any, Callable, Dict, Iterable, Optional, Union

class NeuralCompressor(TrainerModule):
    def init_apply_fn(self):
        """Initialize a default apply function for the model."""
        self.apply_fn = self.model
    def run_model_init(self, exmp_input: Any, init_rng: Any) -> Dict:
        return self.model.init(init_rng, *exmp_input)

class CompressionModel(NDENetwork):
    def MSE(self, x: Any, theta: Any) -> Any:
        return jnp.abs((self(x)-theta)**2).mean()

    def MAE(self, x, theta):
        return jnp.max((self(x)-theta)**2).mean()

class CompressionCNN(CompressionModel):
    num_hiddens : Iterable[int]
    output_size : int
    
    num_images : int
    nside : int
    n_ref : int

    @nn.compact
    def __call__(self, x):
        # (input_number, x, y, image_number, real/imag)
        x = x.reshape((-1, self.nside//self.n_ref, self.nside//self.n_ref, self.num_images, 2))

        x = nn.Conv(features=64, kernel_size=(5, 5, 1), padding='SAME')(x)
        x = nn.max_pool(x, window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(5, 5, 1), padding='SAME')(x)
        x = nn.max_pool(x, window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(5, 5, 1), padding='SAME')(x)
        x = nn.max_pool(x, window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID')
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))

        for num_hidden in self.num_hiddens:
            x = nn.Dense(features=num_hidden)(x)
            x = nn.tanh(x)
        x = nn.Dense(features=self.output_size)(x)
        x = x.reshape((x.shape[0], -1))
        # print(x.shape) # (1, ?)
        return x

class CompressionPowerSpectrum(CompressionModel):
    num_hiddens : Iterable[int]
    output_size : int

    @nn.compact
    def __call__(self, x):
        
        conv = nn.Conv(features=8, kernel_size=(5,), padding='VALID')(x)
        conv = nn.avg_pool(conv, window_shape=(1, 2), strides=(1, 2), padding='VALID')
        conv = nn.relu(conv)
        x = jnp.concatenate([x, conv], axis=1)
        for num_hidden in self.num_hiddens:
            x = nn.Dense(features=num_hidden)(x)
            x = nn.tanh(x)
        x = nn.Dense(features=self.output_size)(x)
        # print(x.shape) # (1, 7)
        return x

class CompressionJoint(CompressionModel):
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
    
def get_simulations(path_output, sims=[0], rot_is=[0], rot_js=[0], noise_reals=1, selected_images=[0]):
    pixels = []
    cls =  []
    parameters = []
    blank_pixels = 0
    n_ref = 1
    indices = product(sims, rot_is, rot_js, range(noise_reals))
    for num_sim, rot_i, rot_j, noise_real in indices:
        output = jnp.load(path_output + f'forward_model_sim{num_sim:05d}_nside0512_rot{rot_i}{rot_j}_noisereal{noise_real}.npy', allow_pickle=True).item()
        nside = output['config']['preprocessing']['nside']

        param = jnp.array([
            output['cosmo_params']["sigma_8"][0],
            output['cosmo_params']["Omega_m"][0], 
            # output['cosmo_params']["Omega_b"][0], 
            # output['cosmo_params']["h"][0], 
            # output['cosmo_params']["n_s"][0],
            # output['cosmo_params']["m_nu"][0],
            # output['cosmo_params']["w"][0],
            # output['cosmo_params']["A_s"][0],
        ])
        parameters.append(param)
        del param

        idx_gal = output['bin_1']['idx']
        shear_map = np.full(12*nside*nside, blank_pixels, dtype=jnp.complex128)
        shear_map[idx_gal] = output['bin_1']['masked_shear_map'] + output['bin_1']['noise_map']
        images = get_cells_from_map(nside, shear_map, n_ref=n_ref)[selected_images]
        images = jnp.moveaxis(images, 0, -1)
        images = jnp.array([images.real, images.imag])
        images = jnp.moveaxis(images, 0, -1)
        # (x, y, image_number, real/imag)
        images = images.flatten()
        pixels.append(images)
        del images

        cl_k, cl_E, cl_B = output["bin_1"]["cl_FS_gamma"][:3]
        cls.append(jnp.concatenate([cl_k, cl_E, cl_B]))
        del cl_k, cl_E, cl_B

        del output
    print(f"Loaded {len(sims)*len(rot_is)*len(rot_js)*noise_reals} simulations")
    return jnp.array(pixels), jnp.array(cls), jnp.array(parameters)

def get_datasets(x, theta, train_test_split=[0.7, 0.2, 0.1]):
    train_test_split = np.array(train_test_split)
    num_sims = x.shape[0]
    key = jr.PRNGKey(np.random.randint(0, 1000))
    index_permutation = jr.permutation(key, num_sims)
    train_fraction, val_fraction, test_fraction = train_test_split/train_test_split.sum()
    train_idx = index_permutation[: int(train_fraction * num_sims)]
    val_idx = index_permutation[
        int(train_fraction * num_sims) : int(
            (train_fraction + val_fraction) * num_sims
        )
    ]
    test_idx = index_permutation[int((train_fraction + val_fraction) * num_sims) :]
    train_data = jdl.ArrayDataset(x[train_idx], theta[train_idx])
    valid_data = jdl.ArrayDataset(x[val_idx], theta[val_idx])
    test_data = jdl.ArrayDataset(x[test_idx], theta[test_idx])
    print(f"Training set: {len(train_data)} simulations.\nValidation set: {len(valid_data)} simulations.\nTest set: {len(test_data)} simulations.")
    return train_data, valid_data, test_data
