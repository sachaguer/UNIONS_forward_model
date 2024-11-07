import os
import numpy as np
import healpy as hp
import pandas as pd
import camb
from scipy.optimize import minimize_scalar
from astropy.table import Table

"""
utils.py
Author: Sacha Guerrini

Utility functions to perform the forward model of the UNIONS shear maps.
"""

rot_footprint_angle = [0, 46, 90, 135, 180]

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
    Returns a dictionary with the cosmologial parameters used to generate the simulation indexed by sim for the Gower Street simulations.

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

def rotate_gals(ras, decs, gammas1, gammas2, rotangle_rad=None, rotangle_dec=None, inv=False, units="deg", rot=None):
    """ Rotates survey patch s.t. its center of mass lies in the origin.
    Author: Pierre Burger
    
    Parameters
    ----------
    ras : np.array
        Right ascension of the galaxies.
    decs : np.array
        Declination of the galaxies.
    gammas1 : np.array
        First component of the shear.
    gammas2 : np.array
        Second component of the shear.
    rotangle_rad : float
        Rotation angle in radians.
    rotangle_dec : float
        Rotation angle in declination in radians.
    inv : bool
        If True, the inverse rotation is applied.
    units : str
        Units of the input angles.

    Returns
    -------
    np.array
        Rotated right ascension.
    np.array
        Rotated declination.
    np.array
        Rotated first component of the shear.
    np.array
        Rotated second component of the shear.
    """
    
    # This is very badly coded TBC
    if rot is None:
        assert rotangle_dec is not None, "rotangle_rad/dec or rot must be specified."
        assert rotangle_rad is not None, "rotangle_rad/dec or rot must be specified."
    elif (rotangle_dec is None) or (rotangle_rad is None):
        assert rot is not None, "rot or rotangle_rad/dec must be specified."
    # Map (ra, dec) --> (theta, phi)
    if units=="deg":
        decs_rad = decs * np.pi/180.
        ras_rad = ras * np.pi/180.
        if rotangle_rad is not None:
            rotangle_rad = rotangle_rad * np.pi/180.
        if rotangle_dec is not None:
            rotangle_dec = rotangle_dec * np.pi/180.
    thetas = np.pi/2. + decs_rad
    phis = ras_rad
    
    # Compute rotation angle
    if rot is None:
        thisrot = hp.Rotator(rot=[-rotangle_dec,rotangle_rad], deg=False, inv=inv)
    else:
        thisrot = rot
    rotatedthetas, rotatedphis = thisrot(thetas,phis, inv=False)

    gamma_rot = (gammas1+1J*gammas2) * np.exp(1J * 2 * thisrot.angle_ref(rotatedthetas, rotatedphis,inv=True))
    
    # Transform back to (ra,dec)
    ra_rot = rotatedphis
    dec_rot = rotatedthetas - np.pi/2.
    if units=="deg":
        dec_rot *= 180./np.pi
        ra_rot *= 180./np.pi
    
    return ra_rot, dec_rot, gamma_rot.real, gamma_rot.imag

def load_sources(path, config):
    """
    Load the source galaxies from the UNIONS from a FITS file and apply rotations to the galaxies to modify the footprint.

    Parameters
    ----------
    path : str
        Path to the FITS file containing the source galaxies.
    config : dict
        Dictionary with the configuration parameters to read the catalog.

    Returns
    -------
    np.array
        Right ascension of the galaxies.
    np.array
        Declination of the galaxies.
    np.array
        First component of the shear.
    np.array
        Second component of the shear.
    np.array
        Weights of the galaxies.
    """

    data = Table.read(path)
    
    rac = data[config['ra_col']]
    dec = data[config['dec_col']]
    weight = data[config['w_col']]
    e_1 = data[config['e1_col']]
    e_2 = data[config['e2_col']]

    gal_2_rot = (rac<50)
    ra_rot,dec_rot,e_1_rot,e_2_rot = rotate_gals(ras=rac[gal_2_rot],decs=dec[gal_2_rot],gammas1=e_1[gal_2_rot], gammas2=e_2[gal_2_rot],rotangle_rad= -20, rotangle_dec= 0)
    dec[gal_2_rot] = dec_rot
    rac[gal_2_rot] = ra_rot
    e_1[gal_2_rot] = e_1_rot
    e_2[gal_2_rot] = e_2_rot

    gal_2_rot = (rac>300)
    ra_rot,dec_rot,e_1_rot,e_2_rot = rotate_gals(ras=rac[gal_2_rot],decs=dec[gal_2_rot],gammas1=e_1[gal_2_rot], gammas2=e_2[gal_2_rot],rotangle_rad= -20, rotangle_dec= 0)
    dec[gal_2_rot] = dec_rot
    rac[gal_2_rot] = ra_rot
    e_1[gal_2_rot] = e_1_rot
    e_2[gal_2_rot] = e_2_rot

    gal_2_rot = (rac<50)
    ra_rot,dec_rot,e_1_rot,e_2_rot = rotate_gals(ras=rac[gal_2_rot],decs=dec[gal_2_rot],gammas1=e_1[gal_2_rot], gammas2=e_2[gal_2_rot],rotangle_rad= 0, rotangle_dec= 50)
    dec[gal_2_rot] = dec_rot
    rac[gal_2_rot] = ra_rot
    e_1[gal_2_rot] = e_1_rot
    e_2[gal_2_rot] = e_2_rot

    gal_2_rot = (dec>20)
    ra_rot,dec_rot,e_1_rot,e_2_rot = rotate_gals(ras=rac[gal_2_rot],decs=dec[gal_2_rot],gammas1=e_1[gal_2_rot], gammas2=e_2[gal_2_rot],rotangle_rad= 0, rotangle_dec= 5)
    dec[gal_2_rot] = dec_rot
    rac[gal_2_rot] = ra_rot
    e_1[gal_2_rot] = e_1_rot
    e_2[gal_2_rot] = e_2_rot

    gal_2_rot = (dec<20)
    ra_rot,dec_rot,e_1_rot,e_2_rot = rotate_gals(ras=rac[gal_2_rot],decs=dec[gal_2_rot],gammas1=e_1[gal_2_rot], gammas2=e_2[gal_2_rot],rotangle_rad= -8.5, rotangle_dec= 0)
    dec[gal_2_rot] = dec_rot
    rac[gal_2_rot] = ra_rot
    e_1[gal_2_rot] = e_1_rot
    e_2[gal_2_rot] = e_2_rot

    gal_2_rot = (dec<20)
    ra_rot,dec_rot,e_1_rot,e_2_rot = rotate_gals(ras=rac[gal_2_rot],decs=dec[gal_2_rot],gammas1=e_1[gal_2_rot], gammas2=e_2[gal_2_rot],rotangle_rad= 0, rotangle_dec= 5)
    dec[gal_2_rot] = dec_rot
    rac[gal_2_rot] = ra_rot
    e_1[gal_2_rot] = e_1_rot
    e_2[gal_2_rot] = e_2_rot
    
    gal_2_rot = (dec<20)
    ra_rot,dec_rot,e_1_rot,e_2_rot = rotate_gals(ras=rac[gal_2_rot],decs=dec[gal_2_rot],gammas1=e_1[gal_2_rot], gammas2=e_2[gal_2_rot],rotangle_rad= -5, rotangle_dec= 0)
    dec[gal_2_rot] = dec_rot
    rac[gal_2_rot] = ra_rot
    e_1[gal_2_rot] = e_1_rot
    e_2[gal_2_rot] = e_2_rot
    
    gal_2_rot = (dec>65)&(rac>0)
    ra_rot,dec_rot,e_1_rot,e_2_rot = rotate_gals(ras=rac[gal_2_rot],decs=dec[gal_2_rot],gammas1=e_1[gal_2_rot], gammas2=e_2[gal_2_rot],rotangle_rad= 5, rotangle_dec= 0)
    dec[gal_2_rot] = dec_rot
    rac[gal_2_rot] = ra_rot
    e_1[gal_2_rot] = e_1_rot
    e_2[gal_2_rot] = e_2_rot
    
    
    return rac,dec,e_1,e_2,weight

def get_rotation(ra, dec, e1, e2, i, j, verbose=False):
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
    e1 : np.array
        First component of the shear.
    e2 : np.array
        Second component of the shear
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
        First component of the shear.
    np.array
        Second component of the shear.
    np.array
        Weights of the galaxies.
    """
    if verbose:
        print(f"[!] Applying a rotation of ({i*360/5}, {rot_footprint_angle[j]}) degrees to the UNIONS footprint.")
    rot1 = hp.Rotator(rot=[0,i*360/5], deg=True, inv=False)
    rot2 = hp.Rotator(rot=[rot_footprint_angle[j], 0], deg=True, inv=False)
    thisrot = rot2 * rot1
    ra, dec, e1, e2 = rotate_gals(ra, dec, e1, e2, rot=thisrot)
    
    return ra, dec, e1, e2