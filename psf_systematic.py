import os
import numpy as np
import scipy.stats as stats
import healpy as hp
from astropy.io import fits

from utils import load_sources, rotate_gals, rot_footprint_angle

"""
psf_systematic.py
Author: Sacha Guerrini

Script to compute systematic maps to be added to shear map to forward model systematics related to the PSF.
"""

def get_rotated_stars(ra, dec, e1_star, e2_star, e1_psf, e2_psf, i, j, verbose=False):
    """
    Get the requested rotation of the UNIONS footprint.
    i = 0, 1, 2, 3, 4 correspond to 5 independant UNIONS footprint
    j = 0, 1, 2, 3, 4 correspond to 5 pseudo-independant rotations of the footprint

    Parameters
    ----------
    ra : np.array
        Right ascension of the galaxies.
    dec : np.array
        Declination of the galaxies.
    e1_star : np.array
        First component of the star ellipticity.
    e2_star : np.array
        Second component of the star ellipticity
    e1_psf : np.array
        First component of the PSF ellipticity.
    e2_psf : np.array
        Second component of the PSF ellipticity.
    i : int
        Index of the footprint.
    j : int
        Index of the rotation.
    verbose : bool
        If True, print information about the rotation.

    Returns
    -------
    np.array
        Right ascension of the galaxies.
    np.array
        Declination of the galaxies.
    np.array
        First component of the star ellipticity.
    np.array
        Second component of the star ellipticity.
    np.array
        First component of the PSF ellipticity.
    np.array
        Second component of the PSF ellipticity.
    """
    if verbose:
        print(f"[!] Applying a rotation of ({i*360/5}, {rot_footprint_angle[j]}) degrees to the UNIONS star footprint.")
    rot1 = hp.Rotator(rot=[0,i*360/5], deg=True, inv=False)
    rot2 = hp.Rotator(rot=[rot_footprint_angle[j], 0], deg=True, inv=False)
    thisrot = rot2 * rot1
    _, _, e1_star, e2_star = rotate_gals(ra, dec, e1_star, e2_star, rot=thisrot)
    ra, dec, e1_psf, e2_psf = rotate_gals(ra, dec, e1_psf, e2_psf, rot=thisrot)
    
    return ra, dec, e1_star, e2_star, e1_psf, e2_psf

def sys_map(psf_cat_path, alpha, beta, eta, nside, config, i=0, j=0):
    """
    Generate a map of systematic error in the ellipticity of galaxies.

    $$
    \delta e^{\rm sys}_{\rm PSF} = \alpha e_{\rm PSF} + \beta (e_* - e_{\rm PSF}) + \eta e_* \delta T
    $$

    Parameters
    ----------
    psf_cat: np.array
        Catalog of the reserved stars containing the PSF and star ellipticities/sizes.
    alpha: float
        Leakage parameter.
    beta: float
        Ellipticity residual parameter.
    eta: float
        Size residual parameter.
    nside: int
        Resolution of the Healpix map.
    config: dict
        Configuration dictionnary containing the name of the columns.
    i: int
        Index of the footprint.
    j: int
        Index of the rotation.
    
    Returns
    -------
    np.array
        Map of systematic error in the ellipticity of galaxies.
    """

    sys_map_e1 = np.zeros(hp.nside2npix(nside))
    sys_map_e2 = np.zeros(hp.nside2npix(nside))

    ra, dec, e1_star, e2_star, e1_psf, e2_psf, size_star, size_psf = load_sources(psf_cat_path, config, cat_type='star')

    ra, dec, e1_star, e2_star, e1_psf, e2_psf = get_rotated_stars(ra, dec, e1_star, e2_star, e1_psf, e2_psf, i, j)
    
    #Create a map of the stars
    theta = (90 - dec) * np.pi / 180
    phi = ra * np.pi / 180
    pix = hp.ang2pix(nside, theta, phi)

    unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)

    n_star = np.zeros(hp.nside2npix(nside))

    n_star[unique_pix] += np.bincount(idx_rep)

    #Compute the systematic error map
    sys_map_e1[unique_pix] += np.bincount(
        idx_rep,
        weights=alpha*e1_psf + beta*(e1_star - e1_psf) + eta*e1_star*(size_star-size_psf)/size_star
    )

    sys_map_e2[unique_pix] += np.bincount(
        idx_rep,
        weights=alpha*e2_psf + beta*(e2_star - e2_psf) + eta*e2_star*(size_star-size_psf)/size_star
    )

    mask = n_star != 0
    sys_map_e1[mask] /= n_star[mask]
    sys_map_e2[mask] /= n_star[mask]

    return sys_map_e1 +1j*sys_map_e2, mask

def sample_sys_map(path_psf_cat, nside, config, prior, verbose, i=0, j=0):
    """
    Sample the systematic error map by sampling the parameters $(\alpha, \beta, \eta)$.

    Parameters
    ----------
    path_psf_cat: str
        Path to the catalog of the reserved stars.
    nside: int
        Resolution of the Healpix map.
    config: dict
        Configuration dictionnary containing the name of the columns.
    prior: dict
        Prior dictionnary containing the mean and the covariance of the parameters.
    verbose: bool
        Whether to print information or not.
    i: int
        Index of the footprint.
    j: int
        Index of the rotation.

    Returns
    -------
    float
        Leakage parameter.
    float
        Ellipticity residual parameter.
    float
        Size residual parameter.
    np.array
        Map of systematic error in the ellipticity of galaxies.
    """
    if not os.path.exists(path_psf_cat):
        raise FileNotFoundError(f"The star catalog {path_psf_cat} does not exist.")
    
    hdu = config['hdu'] #Useless in this version but might be used in the future.

    proposal = stats.multivariate_normal(mean=prior['mean'], cov=prior['cov'])
    alpha, beta, eta = proposal.rvs()

    if verbose:
        print(f"[!] Sampling the systematic error map with alpha={alpha:.4f}, beta={beta:.2f}, eta={eta:.2f}...")
    
    sys_map_, mask = sys_map(path_psf_cat, alpha, beta, eta, nside, config, i, j)
    idx_star = np.arange(len(mask))[mask]

    return alpha, beta, eta, sys_map_[mask], idx_star



