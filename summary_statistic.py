import numpy as np
import healpy as hp
import pymaster as nmt
"""
summary_statistic.py
Author: Sacha Guerrini

Script to compute summary statistics of the shear maps.
Each class/function takes a shear map as input and return a summary statistics.
"""

def get_pseudo_cls(map_, nside, binning='linear', **kwargs):
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
    lmax = 3*nside
    b_lmax = 3*nside-1

    if binning == 'linear':
        step = kwargs.get('lin_step', 10)
        b = nmt.NmtBin.from_nside_linear(nside, step)
    elif binning == 'powspace':
        n_ell_bins = kwargs.get('n_ell_bins', 28)

        ells = np.arange(lmin, lmax+1)

        #Bin in power space
        power = kwargs.get('power',1/2)
        start = np.power(lmin, power)
        end = np.power(lmax, power)
        bins_ell = np.power(np.linspace(start, end, n_ell_bins+1), 1/power).astype(float)

        #Get bandpowers
        bpws = np.digitize(ells.astype(float), bins_ell) - 1
        bpws[0] = 0
        bpws[-1] = n_ell_bins - 1

        b = nmt.NmtBin(ells=ells, bpws=bpws, lmax=b_lmax)

    ell_eff = b.get_effective_ells()

    f_all = nmt.NmtField(mask=(map_ != 0), maps=[map_.real, map_.imag])

    cl_all = nmt.compute_full_master(f_all, f_all, b)

    return ell_eff, cl_all

def get_beam(theta, lmax):
    def top_hat(b, radius):
        return np.where(abs(b) <= radius, 1 / (np.cos(radius) - 1) / (-2 * np.pi), 0)
    t = theta * np.pi / (60 * 180)
    b = np.linspace(0.0, t * 1.2, 10000)
    bw = top_hat(b, t)
    ### plotting the filter
    # plt.plot(b, bw)
    beam = hp.sphtfunc.beam2bl(bw, b, lmax)
    return beam

def smooth_map(theta, lmax, kappa, nside):
    """
    Smooth a convergence map with a top-hat beam.

    Parameters
    ----------
    theta : float
        The smoothing angle in arcmin.
    lmax : int
        The maximum multipole to compute the beam.
    kappa : array
        The convergence map to smooth.
    nside : int
        The resolution of the map.

    Returns
    -------
    array
        The smoothed convergence map.
    """
    beam = get_beam(theta, lmax)
    almkappa = hp.sphtfunc.map2alm(kappa)
    kappa_smooth = hp.sphtfunc.alm2map(hp.sphtfunc.almxfl(almkappa, beam), nside)
    return kappa_smooth