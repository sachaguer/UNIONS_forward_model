import numpy as np
import healpy as hp
import scipy.constants as const
import camb
from cosmology import Cosmology

from astropy.cosmology import z_at_value, FlatLambdaCDM, wCDM
from astropy import units as u

import glass.shells
import glass.lensing

import bornraytrace.lensing as lensing
import bornraytrace.intrinsic_alignments as ia

from tqdm import tqdm

"""
ray_trace.py
Author: Sacha Guerrini

Functions to perform the ray tracing of N-body simulations
"""

def ray_trace_glass(overdensity_array, z_bin_edges, cosmo_params, verbose=False):
    """
    Return the convergence map at each shell for a given cosmology using GLASS.

    Parameters
    ----------
    overdensity_array: np.array
        Multidimensional array with healpy maps at different redshift shells (axis=0)
    z_bin_edges: np.array
        Bin edges of the redshift shells.
    cosmo_params: dict
        Cosmological parameters used to perform the simulation

    Returns
    -------
    np.array
        Weak lensing map for each redshift shell in healpy format with the same resolution than `overdensity_array`.
    """

    nside = hp.npix2nside(overdensity_array.shape[1])

    #Load parameters for the cosmology
    h = cosmo_params["h"]
    Om = cosmo_params["Omega_m"]
    Ob = cosmo_params["Omega_b"]
    Oc = Om - Ob
    ns = cosmo_params["n_s"]
    m_nu = cosmo_params["m_nu"]
    w = cosmo_params["w"]
    As = cosmo_params["A_s"]

    pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
    Onu = pars.omeganu
    Oc = Om - Ob - Onu
    pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)

    lmax = 2*nside #Beyond 2*nside the power spectrum vanishes

    cosmo = Cosmology.from_camb(pars)

    kappa_lensing = np.copy(overdensity_array)*0.

    weights = glass.shells.tophat_windows(z_bin_edges)

    convergence = glass.lensing.MultiPlaneConvergence(cosmo)

    if verbose:
        print("[!] Performing the ray tracing using GLASS...")
        pbar = tqdm(range(overdensity_array.shape[0]))

    else:
        pbar = range(overdensity_array.shape[0])

    for i in pbar:
        convergence.add_window(overdensity_array[i], weights[i]) #Add a convergence plane for then shell considered

        #Compute the convergence field
        kappa_lensing[i] = convergence.kappa

    return kappa_lensing

def ray_trace_bornraytrace(overdensity_array, z_bin_edges, cosmo_params, verbose=False):
    """
    Return the convergence map at each shell for a given cosmology using BornRaytrace code.

    Parameters
    ----------
    overdensity_array: np.array
        Multidimensional array with healpy maps at different redshift shells (axis=0)
    z_bin_edges: np.array
        Bin edges of the redshift shells.
    cosmo_params: dict
        Cosmological parameters used to perform the simulation

    Returns
    -------
    np.array
        Weak lensing map for each redshift shell in healpy format with the same resolution than `overdensity_array`.
    """

    nside = hp.npix2nside(overdensity_array.shape[1])
    lmax = 2*nside #Beyons 2*nside the power spectrum vanishes

    #Load parameters for the cosmology
    h = cosmo_params["h"]
    Om = cosmo_params["Omega_m"]
    Ob = cosmo_params["Omega_b"]
    Oc = Om - Ob
    ns = cosmo_params["n_s"]
    m_nu = cosmo_params["m_nu"]
    w = cosmo_params["w"]
    As = cosmo_params["A_s"]

    pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
    Onu = pars.omeganu
    Oc = Om - Ob - Onu
    pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
    results = camb.get_results(pars)
    comoving_edges = results.comoving_radial_distance(z_bin_edges)
    """ z_centre = np.array(
        [results.redshift_at_comoving_radial_distance((comoving_edges[i+1] + comoving_edges[i])/2) for i in range(len(z_bin_edges)-1)]
    ) """
    z_centre = np.array(
        [(z_bin_edges[i+1]+z_bin_edges[i])*0.5 for i in range(len(z_bin_edges)-1)]
    )
    comoving_edges = comoving_edges * u.Mpc

    kappa_lensing = np.copy(overdensity_array)*0.

    if verbose:
        print("[!] Performing the ray tracing using BornRaytrace...")
        pbar = tqdm(range(1, overdensity_array.shape[0]+1))
    else:
        pbar = range(1, overdensity_array.shape[0]+1)

    for i in pbar:
        kappa_lensing[i-1] = lensing.raytrace(
            100*h[0]*u.km/u.s/u.Mpc, Om,
            overdensity_array = overdensity_array[:i].T,
            a_centre = 1./(1.+z_centre[:i]),
            comoving_edges = comoving_edges[:(i+1)]
        )

    return kappa_lensing

def ray_trace(overdensity_array, z_bin_edges, cosmo_params, method="glass", verbose=False):
    """
    Return the convergence map at each shell for a given cosmology.

    Parameters
    ----------
    overdensity_array: np.array
        Multidimensional array with healpy maps at different redshift shells (axis=0)
    z_bin_edges: np.array
        Bin edges of the redshift shells.
    cosmo_params: dict
        Cosmological parameters used to perform the simulation
    method: str
        Method used to perform the ray tracing. Options are "glass" or "bornraytrace".
    verbose: bool
        If True, print the progress of the ray tracing.

    Returns
    -------
    np.array
        Weak lensing map for each redshift shell in healpy format with the same resolution than `overdensity_array`.
    """

    if method == "glass":
        return ray_trace_glass(overdensity_array, z_bin_edges, cosmo_params, verbose=verbose)
    elif method == "bornraytrace":
        return ray_trace_bornraytrace(overdensity_array, z_bin_edges, cosmo_params, verbose=verbose)
    else:
        raise ValueError("Invalid method. Options are 'glass' or 'bornraytrace'.")


def intrinsic_alignments(overdensity_array, z_bin_edges, cosmo_params, A_ia, eta_ia):
    #Load parameters for the cosmology
    h = cosmo_params["h"]
    Om = cosmo_params["Omega_m"]
    Ob = cosmo_params["Omega_b"]
    Oc = Om - Ob
    ns = cosmo_params["n_s"]
    m_nu = cosmo_params["m_nu"]
    w = cosmo_params["w"]
    As = cosmo_params["A_s"]

    pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
    Onu = pars.omeganu
    Oc = Om - Ob - Onu
    pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
    results = camb.get_background(pars)

    comoving_edges = results.comoving_radial_distance(z_bin_edges)
    z_centre = np.array(
        [(z_bin_edges[i+1]+z_bin_edges[i])*0.5 for i in range(len(z_bin_edges)-1)]
    )

    c1 = (5e-14 * (u.Mpc**3.)/(u.solMass))
    c1_cgs = (c1 * ((1./(cosmo_params["h"][0]))**2.)).cgs
    H0 = 100 * cosmo_params["h"][0] * u.km/u.s/u.Mpc
    H0 = H0.to(u.s**-1)
    G = const.G * u.m**3/(u.kg * u.s**2)
    G = G.cgs
    rho_crit = 3 * H0**2 / (8 * np.pi * G)
    rho_c1 = (c1_cgs * rho_crit).value
    f_nla_array = ia.F_nla(z_centre, Om, A_ia, rho_c1=rho_c1, eta=eta_ia, z0=0.62)
    return (overdensity_array.T * f_nla_array).T
