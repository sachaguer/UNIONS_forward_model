import os
import numpy as np
import scipy.stats as stats
import healpy as hp
from astropy.io import fits

"""
psf_systematic.py
Author: Sacha Guerrini

Script to compute systematic maps to be added to shear map to forward model systematics related to the PSF.
"""

def sys_map(psf_cat, alpha, beta, eta, nside, config):
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
    
    Returns
    -------
    np.array
        Map of systematic error in the ellipticity of galaxies.
    """

    sys_map_e1 = np.zeros(hp.nside2npix(nside))
    sys_map_e2 = np.zeros(hp.nside2npix(nside))

    ra = psf_cat[config['ra_col']]
    dec = psf_cat[config['dec_col']]
    e1_psf = psf_cat[config['e1_psf_col']]
    e2_psf = psf_cat[config['e2_psf_col']]
    size_psf = psf_cat[config['size_psf_col']]**2 if config['square_size'] else psf_cat[config['size_psf_col']]
    e1_star = psf_cat[config['e1_star_col']]
    e2_star = psf_cat[config['e2_star_col']]
    size_star = psf_cat[config['size_star_col']]**2 if config['square_size'] else psf_cat[config['size_star_col']]
    
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

    return sys_map_e1 +1j*sys_map_e2

def sample_sys_map(path_psf_cat, nside, config, prior, verbose):
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
    
    hdu = config['hdu']
    psf_cat = fits.getdata(path_psf_cat, hdu=hdu)

    proposal = stats.multivariate_normal(mean=prior['mean'], cov=prior['cov'])
    alpha, beta, eta = proposal.rvs()

    if verbose:
        print(f"[!] Sampling the systematic error map with alpha={alpha:.4f}, beta={beta:.2f}, eta={eta:.2f}...")

    return alpha, beta, eta, sys_map(psf_cat, alpha, beta, eta, nside, config)



