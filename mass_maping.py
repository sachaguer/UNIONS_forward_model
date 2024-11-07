"""
mass_maping.py

Author: Sacha Guerrini

Script to perform the mass mapping of the UNIONS shear maps.
"""

import numpy as np
import healpy as hp

import bornraytrace.lensing as lensing

import glass.lensing

def kaiser_squire(gamma_map):
    """
    Perform the Kaiser-Squire mass mapping.

    Parameters
    ----------
    gamma_map: np.array
        Shear map in healpy format.

    Returns
    -------
    np.array
        Convergence map in healpy format.
    """
    nside = hp.npix2nside(len(gamma_map))
    kappa = lensing.shear2kappa(gamma_map, lmax=2*nside)
    return -kappa.real, -kappa.imag #Returns E and B modes separately
