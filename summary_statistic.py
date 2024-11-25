import numpy as np
import healpy as hp
import pymaster as nmt
"""
summary_statistic.py
Author: Sacha Guerrini

Script to compute summary statistics of the shear maps.
Each class/function takes a shear map as input and return a summary statistics.
"""

def get_pseudo_cls(map_, nside):
    """
    Compute the pseudo-Cl of a map.
    
    Parameters
    ----------
    map_ : array
        The map to compute the pseudo-Cl from.
    nside : int
        The nside of the map.
        
    Returns
    -------
    array
        The pseudo-Cl of the map.
    """
    lmin = 8
    lmax = 3*nside-1

    b = nmt.NmtBin.from_nside_linear(nside, 10)

    ell_eff = b.get_effective_ells()

    f_all = nmt.NmtField(mask=(map_ != 0), maps=[map_.real, map_.imag])

    cl_all = nmt.compute_full_master(f_all, f_all, b)

    return ell_eff, cl_all