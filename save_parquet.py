import time
start = time.time()

import os
import itertools
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tqdm as tqdm

import pyarrow as pa
import pyarrow.parquet as pq
import healpy as hp
from generate_image import get_cells_from_map
from mass_mapping import kaiser_squire

path_sims = "/lustre/fswork/projects/rech/prk/commun/GowerStreetSims/UNIONS_processing/"
parquet_file = "~/sbi_input/simulations_noisereals2.parquet"

df = pd.DataFrame([])
table = pa.Table.from_pandas(df)
pq.write_table(table, parquet_file)
existing_df = pq.read_table(parquet_file).to_pandas()
print(len(existing_df))

filenames = os.listdir(path_sims)


batch_size = 50
print(f"{len(filenames)} files in directory: {int(np.ceil(len(filenames)/batch_size))} batches")

print(f"ETA: {int((0.0554*len(filenames))//60)} hours {int((0.0554*len(filenames))%60)} minutes")
#Define the schema
schema = pa.schema([
    pa.field('h', pa.float32()),
    pa.field('Omega_m', pa.float32()),
    pa.field('Omega_b', pa.float32()),
    pa.field('sigma_8', pa.float32()),
    pa.field('n_s', pa.float32()),
    pa.field('w', pa.float32()),
    pa.field('m_nu', pa.float32()),
    pa.field('A_s', pa.float32()),
    pa.field('kappa_E', pa.list_(pa.float32())),
    pa.field('kappa_B', pa.list_(pa.float32())),
    pa.field('shape_map', pa.list_(pa.int32()))],
    metadata={
        'h': "Dimensionless Hubble constant",
        'Omega_m': "Matter density",
        'Omega_b': "Baryon density",
        'sigma_8': "Variance of the linear power spectrum in balls of radius 8 Mpc/h",
        'n_s': "Tilt of the primordial power spectrum.",
        "w": "Equation of state of dark energy.",
        "m_nu": "Mass of massive neutrinos.",
        "A_s": "Amplitude of the primordial power spectrum.",
        "kappa_E": "Map of convergence E mode.",
        "kappa_B": "Map of convergence B mode.",
        "shape_map": "Shape of the convergence maps."
    })

#Generator that yields rows as tuple
def get_rows():
    
    for filename in filenames:
        file_path = os.path.join(path_sims, filename)
        if(os.path.isdir(file_path)):
            continue
        if(not "forward_model" in filename):
            print(f"skipped file '{filename}'")
            continue
        try:
            sim = np.load(file_path, allow_pickle=True).item()
            parameters = sim["cosmo_params"]        

            gamma_map = np.zeros(512*512*12, dtype=np.complex64)
            gamma_map[sim['bin_1']['idx']] = sim['bin_1']['masked_shear_map'] + sim['bin_1']['noise_map']

            kappa_E, kappa_B = kaiser_squire(gamma_map)
            
            h = parameters["h"][0]
            Omega_m = parameters["Omega_m"][0]
            Omega_b = parameters["Omega_b"][0]
            sigma_8 = parameters["sigma_8"][0]
            n_s = parameters["n_s"][0]
            w = parameters["w"][0]
            m_nu = parameters["m_nu"][0]
            A_s = parameters["A_s"][0]
    
            patch_selection = [0, 1, 2, 3, 4, 5, 6, 7]
            kappa_E_patches = get_cells_from_map(512, kappa_E, n_ref=1)[patch_selection].flatten()
            kappa_B_patches = get_cells_from_map(512, kappa_B, n_ref=1)[patch_selection].flatten()
            
            shape_map = pa.array((len(patch_selection), 512, 512), type=pa.int32())
            yield (h, Omega_m, Omega_b, sigma_8, n_s, w, m_nu, A_s, kappa_E_patches, kappa_B_patches, shape_map)
        except Exception as e:
            print(f"Cannot convert file '{filename}': {e}")
            continue

# Function to yield batches of data
def get_batches(rows_iterable, chunk_size, schema):
    rows_it = iter(rows_iterable)
    while True:
        batch_df = pd.DataFrame(
            itertools.islice(rows_it, chunk_size),
            columns=schema.names
        )
        if batch_df.empty:
            break  # Stop when no more data
        yield pa.RecordBatch.from_pandas(batch_df, schema=schema, preserve_index=False)

# Create and write Parquet file in batches
rows = get_rows()
batches = get_batches(rows, chunk_size=batch_size, schema=schema)

i=0
with pq.ParquetWriter(parquet_file, schema=schema) as writer:
    for batch in batches:
        i+=1
        try:
            writer.write_batch(batch)
            print(f"Batch {i} done!")
        except Exception as e:
            t = (time.time() - start)/60
            print(f"Failed after {int(t//60)} hours and {int(t%60)} minutes, at batch {i}: {e}")
            raise

t = (time.time() - start)/60
print(f"Parquet file written successfully in {int(t//60)} hours {int(t%60)} minutes!")

print(pd.read_parquet(parquet_file).info())