import os
import numpy as np
import healpy as hp
import pandas as pd
import camb
from scipy.optimize import minimize_scalar

"""
utils.py
Author: Sacha Guerrini

Utility functions to perform the forward model of the UNIONS shear maps.
"""

def downgrade_lightcone(lightcone, nside_out, verbose=False):
    """
    Downgrade a lightcone to a lower resolution (nside_out < nside_in)
    """
    nside_in = hp.npix2nside(len(lightcone))
    if nside_in < nside_out:
        raise ValueError("nside_out must be smaller than nside_in")
    if nside_in == nside_out:
        if verbose:
            print("[!] No downgrade needed.")
        return lightcone
    if verbose:
        print(f"[!] Downgrading lightcone from nside={nside_in} to nside={nside_out}...")
    lightcone_down = hp.ud_grade(lightcone, nside_out)
    if verbose:
        print(f"[!] Done.")
    return lightcone_down

def downgrade_mask(mask, nside_out, verbose=False, threshold=0.5):
    """
    Downgrade a mask to a lower resolution (nside_out < nside_in)
    """
    nside_in = hp.npix2nside(len(mask))
    if nside_in < nside_out:
        raise ValueError("nside_out must be smaller than nside_in")
    if nside_in == nside_out:
        if verbose:
            print("[!] No downgrade needed.")
        return mask
    if verbose:
        print(f"[!] Downgrading mask from nside={nside_in} to nside={nside_out}...")
    mask_down = hp.ud_grade(mask, nside_out)
    mask_down[mask_down < threshold] = 0 #Set to zero the values below 0.5
    mask_down[mask_down >= threshold] = 1 #Set to one the values above 0.5
    if verbose:
        print(f"[!] Done.")
    return mask_down

def apply_mask(map_, mask):
    """
    Apply a binary mask to a map.

    Parameters
    ----------
    map_ : np.array
        Healpix map.
    mask : np.array
        Healpix mask.
    
    Returns
    -------
    np.array
        Masked map.
    """
    nside_map = hp.npix2nside(len(map_))
    nside_mask = hp.npix2nside(len(mask))
    assert nside_map == nside_mask, "The nside of the map and the mask must be the same."
    return map_*mask

def sigma8_difference(logAs, cosmo_params):
    """
    Computes the absolue difference between the sigma8 computed with the input As and the
    target specified in cosmo_params.

    Parameters
    ----------
    logAs : float
        Natural logarithm of the amplitude of the primordial power spectrum.
    cosmo_params : dict
        Dictionary with the cosmological parameters. Does contain the target value of sigma8

    Returns
    -------
    float
        Absolute difference between the computed sigma8 and the target.
    """
    pars = camb.set_params(
        H0=cosmo_params["h"]*100,
        ombh2=cosmo_params["Omega_b"]*cosmo_params["h"]**2,
        omch2=(cosmo_params["Omega_m"]-cosmo_params["Omega_b"])*cosmo_params["h"]**2,
        mnu=cosmo_params["m_nu"],
        w=cosmo_params["w"],
        ns=cosmo_params["n_s"],
        As=np.exp(logAs),
        WantTransfer=True,
        NonLinear=camb.model.NonLinear_both
    )
    Omc = cosmo_params["Omega_m"]-cosmo_params["Omega_b"] - pars.omeganu
    pars = camb.set_params(
        H0=cosmo_params["h"]*100,
        ombh2=cosmo_params["Omega_b"]*cosmo_params["h"]**2,
        omch2=Omc*cosmo_params["h"]**2,
        mnu=cosmo_params["m_nu"],
        w=cosmo_params["w"],
        ns=cosmo_params["n_s"],
        As=np.exp(logAs),
        WantTransfer=True,
        NonLinear=camb.model.NonLinear_both
    )
    results = camb.get_results(pars)
    sigma8 = results.get_sigma8()
    return np.abs(sigma8 - cosmo_params["sigma_8"])

def read_cosmo_params(path_info, sim):
    """
    Returns a dictionary with the cosmologial parameters used to generate the simulation indexed by sim.

    Parameters
    ----------
    sim : int
        Index of the simulation.

    Returns
    -------
    dict
        Dictionary with the cosmological parameters.
    """
    info = pd.read_csv(path_info, header=1)
    cosmo_params = {}
    line =  info.loc[info["Serial Number"]==sim]
    cosmo_params["h"] = line["little_h"].values
    cosmo_params["Omega_m"] = line["Omega_m"].values
    cosmo_params["Omega_b"] = line["Omega_b"].values
    cosmo_params["sigma_8"] = line["sigma_8"].values
    cosmo_params["n_s"] = line["n_s"].values
    cosmo_params["w"] = line["w"].values
    cosmo_params["m_nu"] = line["m_nu"].values

    #Compute the value of A_s knowing the value of sigma_8
    res = minimize_scalar(sigma8_difference, args=(cosmo_params,), bracket=[np.log(1e-9), np.log(3e-9)], tol=1e-10)
    cosmo_params["A_s"] = np.exp(res.x)
    return cosmo_params

def get_path_lightcone(path_sims, sim, run):
    path = path = os.path.expanduser(path_sims + "sim{:05d}/".format(sim) + "run.{:05d}.lightcone.npy".format(run))
    return os.path.exists(path), path

def get_path_redshifts(path_sims, sim):
    path = os.path.expanduser(path_sims + "sim{:05d}/".format(sim) + "z_values.txt")
    return os.path.exists(path), path

def apply_random_rotation(e1, e2):
    """
    Apply a random rotation to the ellipticity components e1 and e2.

    Parameters
    ----------
    e1 : np.array
        First component of the ellipticity.
    e2 : np.array
        Second component of the ellipticity.

    Returns
    -------
    np.array
        First component of the rotated ellipticity.
    np.array
        Second component of the rotated ellipticity.
    """
    np.random.seed()
    rot_angle = np.random.rand(len(e1))*2*np.pi
    e1_out = e1*np.cos(rot_angle) + e2*np.sin(rot_angle)
    e2_out = -e1*np.sin(rot_angle) + e2*np.cos(rot_angle)
    return e1_out, e2_out