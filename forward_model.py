import os
import time

import numpy as np
import healpy as hp
import pandas as pd

import glass.shells

from tqdm import tqdm

from ray_trace import ray_trace, intrinsic_alignments
from utils import read_cosmo_params, get_path_lightcone, get_path_redshifts, downgrade_lightcone

"""
forward_model.py
Author: Sacha Guerrini

Script to perform the forward model of the UNIONS shear maps.
"""

def preprocessing_gower_street(path_sims, path_infos, sim_number, nside, nside_intermediate=None, verbose=False):
    """
    Preprocess the Gower Street simulations to get the convergence maps.

    Parameters
    ----------
    path_sims: str
        Path to the Gower Street simulations.
    path_infos: str
        Path to the information files.
    sim_number: int
        Index of the simulation.
    nside: int
        Resolution of the output maps.
    nside_intermediate: int
        Resolution of the intermediate maps. If None, no intermediate map is generated.
    verbose: bool
        If True, print information about the process.

    Returns
    -------
    np.array
        Weak lensing map for each redshift shell in healpy format.
    np.array
        Redshift shells edges
    dict
        Cosmological parameters used to perform the simulation
    """
    assert os.path.exists(path_infos), "The path to the information file does not exist."

    if verbose:
        print(f"[!] Preprocessing the Gower Street simulation {sim_number}...")
        print(f"[!] Reading the cosmological parameters...")
    #Read the cosmological parameters
    start = time.time()
    cosmo_params = read_cosmo_params(path_infos, sim_number)
    if verbose:
        print(f"[!] Done in {(time.time()-start)/60:.2f} min.")

    #Get the path to redshift information
    path_exist, path_redshift = get_path_redshifts(path_sims, sim_number)
    assert path_exist, f"The path to the redshift file {path_redshift} does not exist."

    infos_redshift = pd.read_csv(path_redshift, sep=",")

    overdensity_array = []
    z_bin_edges = []

    if verbose:
        print(f"[!] Extracting overdensity maps and redshift edges for the Gower Street simulation {sim_number}...")
        pbar = tqdm(zip(infos_redshift["# Step"], infos_redshift["z_far"]))
    else:
        pbar = zip(infos_redshift["# Step"], infos_redshift["z_far"])

    start = time.time()
    for step, z_far in pbar:

        path_exist, path_lightcone = get_path_lightcone(path_sims, sim_number, step)

        if path_exist:
            lightcone = np.load(path_lightcone)
            density_i = lightcone/np.mean(lightcone) - 1
            del lightcone
            if nside_intermediate is not None:
                density_i = downgrade_lightcone(density_i, nside_intermediate, verbose=False)
            density_i = downgrade_lightcone(density_i, nside, verbose=False)
            overdensity_array.append(density_i)

            z_bin_edges.append(z_far)
    z_bin_edges.append(0.0)

    z_bin_edges = np.array(z_bin_edges[::-1])
    overdensity_array = np.array(overdensity_array[::-1]) #reverse the array as we read from the larger redshift to the smaller

    if verbose:
        print(f"[!] Done in {(time.time()-start)/60:.2f} min.")
        print("[!] Number of redshift shells:", len(z_bin_edges)-1)
        print("[!] Larger redshift:", z_bin_edges[-1])

    return overdensity_array, z_bin_edges, cosmo_params

def weight_map_w_redshift(map_, z_bin_edges, redshift_distribution, verbose=False):
    """
    Weight a map with a given redshift distribution.

    Parameters
    ----------
    map: np.array
        Healpy map to be weighted with redshift bins represented by z_bin_edges.
    z_bin_edges: np.array
        Bin edges of the redshift shells.
    redshift_distribution: (np.array, np.array)
        Redshift distribution to be used as weight. The tuple contains dndz and z.
    verbose: bool
        If True, print the progress of the weighting.

    Returns
    -------
    np.array
        Weighted map.
    """

    dndz, z = redshift_distribution

    weights = glass.shells.tophat_windows(z_bin_edges)

    map_bar = np.zeros_like(map_[0])

    if verbose:
        pbar = tqdm(range(len(z_bin_edges)-1))
    else:
        pbar = range(len(z_bin_edges)-1)

    for i in pbar:
        z_i, dndz_i = glass.shells.restrict(z, dndz, weights[i])

        ngal = np.trapz(dndz_i, z_i)
        map_bar += ngal * map_[i]

    return map_bar/np.trapz(dndz, z)

def kappa2shear(kappa_map, lmax, verbose=False):
    if verbose:
        print(f"[!] Converting convergence map to shear maps with lmax={lmax}...")
    
    gamma = np.zeros_like(kappa_map, dtype=np.complex64)

    if verbose:
        pbar = tqdm(range(len(kappa_map)))
    else:
        pbar = range(len(kappa_map))

    for i in pbar:
        gamma[i] = glass.lensing.from_convergence(kappa_map[i], lmax=lmax, shear=True)[0]
    
    return gamma


def forward(path_sims, path_infos, sim_name='GowerStreet', add_ia=False, verbose=False, **kwargs):
    """
    Perform the forward model of the UNIONS shear maps.

    Parameters
    ----------
    path_sims: str
        Path to the simulations.
    path_infos: str
        Path to the information files.
    sim_name: str
        Name of the simulation. Currenlty, only 'GowerStreet' is supported.
    verbose: bool
        If True, print information about the process.

    Returns
    -------
    np.array
        Weak lensing map for each redshift shell in healpy format. !!might change!!
    """
    assert sim_name in ['GowerStreet'], "Invalid simulation name. Only 'GowerStreet' is supported."

    if sim_name == 'GowerStreet':
        #Preprocess the Gower Street simulations
        sim_number = kwargs['sim_number']
        nside = kwargs['nside']
        lmax = 2*nside
        nside_intermediate = kwargs.get('nside_intermediate', None)
        overdensity_array, z_bin_edges, cosmo_params = preprocessing_gower_street(path_sims, path_infos, sim_number, nside, nside_intermediate, verbose=verbose)

    method = kwargs.get('method', 'glass')
    #Perform the ray tracing
    kappa_lensing = ray_trace(overdensity_array, z_bin_edges, cosmo_params, method=method, verbose=verbose)

    gamma_lensing = kappa2shear(kappa_lensing, lmax=lmax, verbose=verbose)

    if add_ia:
        A_ia = kwargs.get('A_ia', None)
        eta_ia = kwargs.get('eta_ia', None)
        assert A_ia is not None, "The amplitude of the intrinsic alignment model is not defined."
        assert eta_ia is not None, "The slope of the intrinsic alignment model is not defined."
        kappa_ia = intrinsic_alignments(overdensity_array, z_bin_edges, cosmo_params, A_ia, eta_ia)

        gamma_ia = kappa2shear(kappa_ia, lmax=lmax, verbose=verbose)

        if verbose:
            print("[!] Adding the intrinsic alignment to the shear maps...")
        gamma_lensing += gamma_ia
    return kappa_lensing, kappa_ia, gamma_lensing

def apply_mask(map_, mask, verbose=False):
    """
    Apply a mask to a map.

    Parameters
    ----------
    map_: np.array
        Healpy map to be masked.
    mask: np.array
        Healpy map with the mask.
    verbose: bool
        If True, print information about the process.

    Returns
    -------
    np.array
        Masked map.
    """
    if verbose:
        print("[!] Applying the mask to the map...")

    map_masked = np.copy(map_)

    map_masked[mask == 0] = hp.UNSEEN

    return map_masked