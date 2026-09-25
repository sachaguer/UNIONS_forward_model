import time
start = time.time()

import os
import itertools
import numpy as np

import tqdm as tqdm
import re

import healpy as hp
from summary_statistic import get_pseudo_cls
import datasets
from datasets import Dataset

path_sims = "/lustre/fswork/projects/rech/prk/commun/GowerStreetSims/UNIONS_processing/"
parquet_file = "/lustre/fswork/projects/rech/prk/commun/GowerStreetSims/UNIONS_sims_pseudo_cell.parquet"

filenames = os.listdir(path_sims)


batch_size = 200
num_proc = 20
print(f"{len(filenames)} files in directory: {int(np.ceil(len(filenames)/batch_size))} batches")

min_per_sim = 2.3/60
print(f"ETA: {int((min_per_sim*len(filenames))//60)} hours {int((min_per_sim*len(filenames))%60)} minutes")

datasets.config.DEFAULT_MAX_BATCH_SIZE = batch_size

def get_sim_id(filename):
    # file format: 'forward_model_sim00000_nside0000_rot00_noisereal0.npy'
    pattern = r"sim(\d+)_nside(\d+)_rot(\d+)_noisereal(\d+)"
    match = re.search(pattern, filename)
    if match:
        sim, nside, rot, noisereal = map(int, match.groups())
        return sim*10**7 + nside*10**3 + rot*10 + noisereal
    else:
    	raise ValueError(f"Filename '{filename}' does not match expected pattern.")

def get_items(filenames_chunk):
    for chunk in filenames_chunk:
        for filename in chunk:
            file_path = os.path.join(path_sims, filename)
            if(os.path.isdir(file_path)):
                continue
            if(not "forward_model" in filename):
                print(f"skipped file '{filename}'")
                continue
            try:
                sim = np.load(file_path, allow_pickle=True).item()
                parameters = sim["cosmo_params"]        

                h = parameters["h"][0]
                Omega_m = parameters["Omega_m"][0]
                Omega_b = parameters["Omega_b"][0]
                sigma_8 = parameters["sigma_8"][0]
                n_s = parameters["n_s"][0]
                w = parameters["w"][0]
                m_nu = parameters["m_nu"][0]
                A_s = parameters["A_s"][0]
                sim_id = get_sim_id(filename)

                nside = 512

                gamma_map = np.zeros(hp.nside2npix(nside), dtype=np.complex128)
                gamma_map[sim['bin_1']['idx']] = sim['bin_1']['masked_shear_map'] + sim['bin_1']['noise_map']

            
                lmax = 2*nside
                ell_eff, pseudo_cell = get_pseudo_cls(gamma_map, nside, binning='powspace', lmax=lmax, wsp="workspace_UNIONS.fits")
                ell_eff = ell_eff.astype(np.float32)
                pseudo_cell = pseudo_cell.astype(np.float32)

                
                gamma_map[sim['bin_1']['idx_star']] += sim['bin_1']['sys_map']

                lmax = 2*nside
                ell_eff, pseudo_cell_sys = get_pseudo_cls(gamma_map, nside, binning='powspace', lmax=lmax, wsp="workspace_UNIONS_sys.fits")
                ell_eff = ell_eff.astype(np.float32)
                pseudo_cell_sys = pseudo_cell_sys.astype(np.float32)
                
            
                yield {
                    "id": sim_id,
                    "h": h,
                    "Omega_m": Omega_m, 
                    "Omega_b": Omega_b, 
                    "sigma_8": sigma_8, 
                    "n_s": n_s, 
                    "w": w, 
                    "m_nu": m_nu, 
                    "A_s": A_s, 
                    "ell_eff": ell_eff.flatten(),
                    "pseudo_cell": pseudo_cell.flatten(),
                    "pseudo_cell_sys": pseudo_cell_sys.flatten()
                }
            except Exception as e:
                print(f"Cannot convert file '{filename}': {e}")
                continue

file_chunks = np.array_split(filenames, num_proc)
print("Generating dataset")
ds = Dataset.from_generator(get_items, gen_kwargs={"filenames_chunk": file_chunks}, num_proc=num_proc)
print(ds)

print("Saving dataset")
ds.to_parquet(parquet_file)


t = (time.time() - start)/60
print(f"Parquet file written successfully in {int(t//60)} hours {int(t%60)} minutes!")
