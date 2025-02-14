import os
import time
import h5py

import numpy as np
import healpy as hp
import pandas as pd

import glass.shells

from tqdm import tqdm

from ray_trace import ray_trace, intrinsic_alignments
from utils import read_cosmo_params, get_path_lightcone, get_path_redshifts, downgrade_lightcone, apply_random_rotation

"""
forward_model.py
Author: Sacha Guerrini

Script to perform the forward model of the UNIONS shear maps.
Includes preprocessing function specific to each simulation output.
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

def preprocessing_cosmogrid(path_sims, path_infos, nside, nside_intermediate=None, verbose=False):
    """
    Preprocess the Cosmogrid simulations to get the convergence maps.

    Parameters
    ----------
    path_sims: str
        Path to the Cosmogrid simulations.
    path_infos: str
        Path to the information files.
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
    #!!!not yet correctly implemented: few thingsd are hardcoded!!!

    assert os.path.exists(path_infos), "The path to the information file does not exist."

    meta_info = h5py.File(path_infos, 'r')
    if verbose:
        print(f"[!] Preprocessing the CosmoGridsV1 simulation...")
        print(f"[!] Reading the cosmological parameters...")
    #Read the cosmological parameters
    start = time.time()
    cosmo_params = {}
    cosmo_params['Omega_m'] = meta_info['parameters']['fiducial']['Om'][0]
    cosmo_params['Omega_b'] = meta_info['parameters']['fiducial']['Ob'][0]
    cosmo_params['h'] = meta_info['parameters']['fiducial']['H0'][0]/100
    cosmo_params['n_s'] = meta_info['parameters']['fiducial']['ns'][0]
    cosmo_params['sigma8'] = meta_info['parameters']['fiducial']['s8'][0]/np.sqrt(cosmo_params['Omega_m']/0.3)
    cosmo_params['w'] = meta_info['parameters']['fiducial']['w0'][0]
    cosmo_params['m_nu'] = meta_info['parameters']['fiducial']['m_nu'][0]
    cosmo_params['As'] = meta_info['parameters']['fiducial']['As'][0]

    #Get the overdensity array and shell information
    if verbose:
        print(f"[!] Extracting overdensity maps and redshift edges for the CosmoGridsV1 simulation...")
    path_ = os.path.join(path_sims, meta_info['parameters']['fiducial']['path_par'][0].decode('utf-8'))
    path_ = os.path.join(path_, 'run_0000/compressed_shells.npz') #hardcoded
    compressed_shells = np.load(path_)
    overdensity_array = compressed_shells['shells']
    overdensity_array = overdensity_array/np.mean(overdensity_array, axis=1)[:, None] - 1
    if nside != hp.npix2nside(len(overdensity_array[0])):
        overdensity_array_ = overdensity_array.copy()
        overdensity_array = []
        for i in tqdm(range(len(overdensity_array_))):
            map_ = overdensity_array_[i]
            if nside_intermediate is not None:
                map_ = downgrade_lightcone(map_, nside_intermediate, verbose=False)
            map_ = downgrade_lightcone(map_, nside, verbose=False)
            overdensity_array.append(map_)
        overdensity_array = np.array(overdensity_array)
    shell_info = compressed_shells['shell_info']
    del compressed_shells

    z_bin_edges = np.concatenate((shell_info['lower_z'], [shell_info['upper_z'][-1]]))
    
    if verbose:
        print(f"[!] Done in {(time.time()-start)/60} minutes...")  
        print("[!] Number of redshift shells:", len(z_bin_edges)-1)
        print("[!] Larger redshift:", z_bin_edges[-1])

    return overdensity_array, z_bin_edges, cosmo_params


def weight_map_w_redshift(map_, z_bin_edges, redshift_distribution, bias=0.0, overdensity_array=None, verbose=False):
    """
    Weight a map with a given redshift distribution. For now source clustering is not
    taken into account.

    Parameters
    ----------
    map: np.array
        Healpy map to be weighted with redshift bins represented by z_bin_edges.
    z_bin_edges: np.array
        Bin edges of the redshift shells.
    redshift_distribution: (np.array, np.array)
        Redshift distribution to be used as weight. The tuple contains dndz and z.
    bias: float
        Galaxy bias parameter to be used to compute the source clustering. If 0.0, no source clustering is applied.
    overdensity_array: np.array
        Overdensity array to compute the source clustering. If None, no source clustering is applied.
    verbose: bool
        If True, print the progress of the weighting.

    Returns
    -------
    np.array
        Weighted map.
    """

    if bias != 0.0:
        assert overdensity_array is not None, "Overdensity array is required to compute the source clustering."

    dndz, z = redshift_distribution

    weights = glass.shells.tophat_windows(z_bin_edges)

    map_bar = np.zeros_like(map_[0])

    if verbose:
        pbar = tqdm(range(len(z_bin_edges)-1))
    else:
        pbar = range(len(z_bin_edges)-1)

    normalization_src_clustering = 0 #Add src clustering (See Gatti et al. 2024)
    normalization_dndz = 0
    for i in pbar:
        z_i, dndz_i = glass.shells.restrict(z, dndz, weights[i])

        ngal = np.trapz(dndz_i, z_i)

        if bias != 0.0:
            src_clustering = (1. + bias * overdensity_array[i])
        else:
            src_clustering = 1.0
        map_bar += ngal * map_[i] * src_clustering
        normalization_src_clustering += ngal * src_clustering
        normalization_dndz += ngal
        noise_factor = np.sqrt(normalization_dndz/normalization_src_clustering)

    return map_bar/normalization_src_clustering, noise_factor

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


def forward(path_sims, path_infos, sim_name='GowerStreet', verbose=False, **kwargs):
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
        
    return kappa_lensing, gamma_lensing, overdensity_array, z_bin_edges, cosmo_params

def add_intrinsic_alignment(gamma_lensing, A_ia, eta_ia, overdensity_array, z_bin_edges, cosmo_params, verbose=False):
    """
    Add intrinsic alignment to the shear maps.

    Parameters
    ----------
    gamma_lensing: np.array
        Shear maps.
    A_ia: float
        Amplitude of the intrinsic alignment model.
    eta_ia: float
        Slope of the intrinsic alignment model.
    overdensity_array: np.array
        Overdensity maps.
    z_bin_edges: np.array
        Redshift bin edges.
    cosmo_params: dict
        Cosmological parameters.
    verbose: bool
        If True, print information about the process.
    
    Returns
    -------
    np.array
        Shear maps with intrinsic alignment.
    """
    lmax = 2*hp.get_nside(gamma_lensing[0])
    if verbose:
        print(f"[!] Amplitude of the intrinsic alignment model: {A_ia}")
        print(f"[!] Slope of the intrinsic alignment model: {eta_ia}")
    kappa_ia = intrinsic_alignments(overdensity_array, z_bin_edges, cosmo_params, A_ia, eta_ia)

    gamma_ia = kappa2shear(kappa_ia, lmax=lmax, verbose=verbose)
    del kappa_ia

    gamma_lensing += gamma_ia
    del gamma_ia

    return gamma_lensing



def add_shape_noise(shear_map, ra, dec, e1, e2, w):
    """
    Add shape noise to a map.

    Parameters
    ----------
    shear_map: np.array
        Healpy shear map to add shape noise.
    ra: np.array
        Right ascension of the observed galaxy catalog.
    dec: np.array
        Declination of the observed galaxy catalog.
    e1: np.array
        First component of the ellipticity of the observed galaxy catalog.
    e2: np.array
        Second component of the ellipticity of the observed galaxy catalog.
    w: np.array
        Weights of the observed galaxy.

    Returns
    -------
    np.array
        Masked shear map
    np.array
        Noise map
    np.array
        Pixel indices of the unmasked pixels.
    """
    nside = hp.npix2nside(len(shear_map))

    #Computes a map of the weighted number of galaxies in each pixel
    theta = (90-dec)*np.pi/180
    phi = ra*np.pi/180
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    del theta, phi
    
    n_map = np.zeros(hp.nside2npix(nside))

    #Get a map between pixels in Healpix and galaxies in the galaxy catalog
    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)

    n_map[unique_pix] += np.bincount(idx_rep, weights=w) #weighted number of galaxies per pixel

    #Mask corresponds to pixels with at least one galaxy
    mask = (n_map != 0)

    #Apply random rotations to e1/2 to erase cosmological signal
    e1_rot, e2_rot = apply_random_rotation(e1, e2)
    del e1, e2

    #noise maps
    noise_map_e1 = np.zeros(hp.nside2npix(nside))
    noise_map_e2 = np.zeros(hp.nside2npix(nside))
    #masked shear maps
    g1_map = np.zeros(hp.nside2npix(nside))
    g2_map = np.zeros(hp.nside2npix(nside))

    #Compute the noise maps with the randomly rotated galaxies
    noise_map_e1[unique_pix] += np.bincount(idx_rep, weights=w*e1_rot)
    noise_map_e2[unique_pix] += np.bincount(idx_rep, weights=w*e2_rot)
    noise_map_e1[mask] /= n_map[mask]
    noise_map_e2[mask] /= n_map[mask]

    #Compute the masked shear maps
    g1_ = shear_map.real[pix] #Project on the pixels where objects are located
    g2_ = shear_map.imag[pix]

    g1_map[unique_pix] += np.bincount(idx_rep, weights=w*g1_)
    g2_map[unique_pix] += np.bincount(idx_rep, weights=w*g2_)
    g1_map[mask] /= n_map[mask]
    g2_map[mask] /= n_map[mask]
    #!!!No weighting is applied as for each pixel g1_ is constant contrary to the case of the noise map!!!

    masked_shear_map = g1_map + 1j*g2_map
    noise_map = noise_map_e1 + 1j*noise_map_e2
    idx_ = np.arange(len(mask))[mask]

    return masked_shear_map[mask], noise_map[mask], idx_

def get_reduced_shear(gamma_lensing, kappa_lensing):
    """
    Get the reduced shear from the shear maps.

    $$g = \frac{\gamma}{1-\kappa}$$

    Parameters
    ----------
    gamma_lensing: np.array
        Shear maps.
    kappa_lensing: np.array
        Convergence maps.
    """
    assert len(gamma_lensing) == len(kappa_lensing), "The number of redshift shells in the shear and convergence maps must be the same."

    return gamma_lensing/(1-kappa_lensing)