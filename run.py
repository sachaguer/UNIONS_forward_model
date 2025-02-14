import os
import yaml
import argparse
import time
import copy
import gc

import numpy as np
import scipy.stats as stats
import healpy as hp
import matplotlib.pyplot as plt
import camb
from astropy.io import fits

from forward_model import forward, weight_map_w_redshift, add_shape_noise, add_intrinsic_alignment, get_reduced_shear
from psf_systematic import sample_sys_map
from utils import rot_footprint_angle, load_sources, get_rotation

plt.rcParams.update({
    "axes.labelsize": 14,  # Adjust as needed
    "axes.titlesize": 16,  # Adjust as needed
    "xtick.labelsize": 14,  # Adjust as needed
    "ytick.labelsize": 14,  # Adjust as needed
    "legend.fontsize": 12,  # Adjust as needed
    "figure.figsize": (15, 8),  # Adjust as needed
    "figure.dpi": 300  # Adjust as needed
})

def plot_mollview(map, title, path_output, cmap):
    hp.mollview(map, title=title, cmap=cmap)
    plt.savefig(path_output)
    plt.close()

def validation_plot_dndz(redshift_distr, nbins):
    z = redshift_distr[:, 0]
    
    plt.figure()

    for i in range(1, nbins+1):
        plt.plot(z, redshift_distr[:, i], label=f'Bin {i}')
    plt.xlabel(r'$z$')
    plt.ylabel(r'$n(z)$')
    plt.legend()
    plt.savefig(path_output+'/Plots/dndz.png')
    plt.close()

def validate_convergence_power_spectrum(kappa_lensing, z_bin_edges, redshift_distr, verbose, nside, cosmo_params):

    dndz, z = redshift_distr
    kappa_lensing_bar, _ = weight_map_w_redshift(kappa_lensing, z_bin_edges, (dndz, z), verbose=verbose)

    plot_mollview(kappa_lensing_bar, title='Convergence map weighted by the redshift distribution', path_output=path_output+'/Plots/kappa_lensing_bar.png', cmap='inferno')

    #Compute the power spectrum for validation
    #Load parameters for the cosmology
    h = cosmo_params["h"]
    Om = cosmo_params["Omega_m"]
    Ob = cosmo_params["Omega_b"]
    Oc = Om - Ob
    ns = cosmo_params["n_s"]
    m_nu = cosmo_params["m_nu"]
    w = cosmo_params["w"]
    As = cosmo_params["A_s"]

    lmax = 2*nside

    if verbose:
        print("[!] Computing the power spectrum for validation...")
    pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)
    Oc = Om - Ob - pars.omeganu
    pars = camb.set_params(H0=100*h, omch2=Oc*h**2, ombh2=Ob*h**2, ns=ns, mnu=m_nu, w=w, As=As, WantTransfer=True, NonLinear=camb.model.NonLinear_both)

    #get the angular power spectra of the lensing map
    sim_cls = hp.anafast(kappa_lensing_bar, pol=True, lmax=lmax, use_pixel_weights=True)

    #getthe expected cl's from CAMB
    pars.min_l = 1
    pars.set_for_lmax(lmax)
    pars.SourceWindows = [
        camb.sources.SplinedSourceWindow(z=z, W=dndz, source_type='lensing')
    ]
    theory_cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

    #Plot the power spectra
    plt.figure()

    #get the HEALPix pixel window function since the lensing fields have it
    pw = hp.pixwin(nside, lmax=lmax)

    l = np.arange(lmax+1)
    plt.plot(l, sim_cls, label="simulation", c='k')
    plt.plot(l, theory_cls['W1xW1']*pw**2, label="expectation", c='r')

    plt.xscale('log')
    plt.yscale('log')

    plt.legend()

    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$C_\ell$")
    plt.title("Convergence power spectrum")

    plt.savefig(path_output+'Plots/power_spectrum.png')
    plt.close()

def print_cosmo_params(cosmo_params):
    print("[!] The cosmological parameters are:")
    print(f"[!] h = {cosmo_params['h']}")
    print(f"[!] Omega_m = {cosmo_params['Omega_m']}")
    print(f"[!] Omega_b = {cosmo_params['Omega_b']}")
    print(f"[!] m_nu = {cosmo_params['m_nu']}")
    print(f"[!] w = {cosmo_params['w']}")
    print(f"[!] A_s = {cosmo_params['A_s']}")
    print(f"[!] n_s = {cosmo_params['n_s']}")
    print(f"[!] sigma_8 = {cosmo_params['sigma_8']}")

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", help="Path to the configuration file.", type=str, default='config.yaml')

if __name__ == '__main__':
    print("[!] Starting the forward model...")
    start = time.time()
    # Load the configuration file
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    verbose = True if config['verbose'] == 'T' else False
    validation_plot = config['validation_plot']

    path_output = config['simulation']['path_output']
    output = {}

    #Check if output directory exists
    if not os.path.exists(path_output):
        if verbose:
            print(f"[!] Creating the output directory {path_output}.")
        os.makedirs(path_output)

    if validation_plot == 'T' and not os.path.exists(path_output+'/Plots/'):
        if verbose:
            print(f"[!] Creating the validation plots directory {path_output}/Plots.")
        os.makedirs(path_output+'/Plots/')

    # Perform the forward model
    path_sims = config['simulation']['path_simulation']
    path_info = config['simulation']['path_info']
    sim_idx = config['simulation']['sim_number']
    sim_name = config['simulation']['sim_name']
    ray_tracing_method = config['ray_tracing']['method']
    nside = config['preprocessing']['nside']
    nside_intermediate = config['preprocessing'].get('nside_intermediate', None)
    add_ia = config['intrinsic_alignment']['add_ia']
    reduced_shear = config['ray_tracing']['reduced_shear']

    if verbose:
        print("[!] Performing the forward model...")
        print(f"[!] The chosen resolution is nside={nside} which corresponds to {hp.nside2resol(nside, arcmin=True):.2f} arcmin.")

    #Get the shear map after ray tracing from the Gower Street simulations
    kappa_lensing, gamma_lensing, overdensity_array, z_bin_edges, cosmo_params = forward(
        path_sims, path_info, sim_name=sim_name, sim_number=sim_idx, nside=nside, nside_intermediate=nside_intermediate, method=ray_tracing_method, verbose=verbose
    )
    output['cosmo_params'] = cosmo_params
    np.save(path_output+f'/kappa_lensing_sim{sim_idx:05d}_nside{nside:04d}.npy', kappa_lensing)
    np.save(path_output+f'/overdensity_array_sim{sim_idx:05d}_nside{nside:04d}.npy', overdensity_array)
    np.save(path_output+f'/gamma_lensing_sim{sim_idx:05d}_nside{nside:04d}.npy', gamma_lensing)

    del overdensity_array
    del kappa_lensing
    del gamma_lensing


    if verbose:
        print_cosmo_params(cosmo_params)
    
    output['config'] = config
    
    n_noise_real = config.get('n_noise_real', 0)
    rot_ra = config.get('rot_ra', [0])
    rot_dec = config.get('rot_dec', [0])

    #galaxy bias for source clustering
    b_sc = config['redshift_distribution'].get('b_sc', 0.0)
    if verbose:
        if b_sc == 0.0:
            print("[!] Source clustering is not taken into account.")
        else:
            print(f"[!] Source clustering is taken into account with a bias of b_g={b_sc}.")

    assert all(rot in range(0, 5) for rot in rot_ra), "The rotation in RA must be between 0 and 4."
    assert all(rot in range(0, 5) for rot in rot_dec), "The rotation in DEC must be between 0 and 4."

    #Iterate on the different rotations of the footprint + noise realisation 
    for noise_real in range(n_noise_real): #Two different noise realizations
        for j in rot_ra: #Index for the rotation of the footprint in RA
            for k in rot_dec: #Index for the roation of the footprint in DEC
                
                if verbose:
                    print(f"[!] Performing the forward model for the rotation of {j*360/5} degrees in RA and {rot_footprint_angle[k]} degrees in DEC...")
                start_ = time.time()
                output_ = copy.deepcopy(output)
                nuisance_parameters = {}
                gamma_lensing = np.load(path_output+f'/gamma_lensing_sim{sim_idx:05d}_nside{nside:04d}.npy')
                if add_ia == 'T':
                    A_ia = np.random.uniform(low=config['intrinsic_alignment']['prior_A_ia'][0], high=config['intrinsic_alignment']['prior_A_ia'][1])
                    eta_ia = np.random.uniform(low=config['intrinsic_alignment']['prior_eta_ia'][0], high=config['intrinsic_alignment']['prior_eta_ia'][1])
                    if verbose:
                        print("[!] Adding the intrinsic alignment to the shear maps...")
                    nuisance_parameters['A_ia'] = A_ia
                    nuisance_parameters['eta_ia'] = eta_ia
                    overdensity_array = np.load(path_output+f'/overdensity_array_sim{sim_idx:05d}_nside{nside:04d}.npy')
                    gamma_lensing = add_intrinsic_alignment(gamma_lensing, A_ia, eta_ia, overdensity_array, z_bin_edges, cosmo_params, verbose=verbose)
                    del overdensity_array

                if reduced_shear == 'T':
                    if verbose:
                        print("[!] Compute the reduced shear from the shear and kappa maps...")
                    kappa_lensing = np.load(path_output+f'/kappa_lensing_sim{sim_idx:05d}_nside{nside:04d}.npy')
                    gamma_lensing = get_reduced_shear(gamma_lensing, kappa_lensing)
                    del kappa_lensing

                #Saves the output if required
                save_ray_tracing = config['ray_tracing']['save_ray_tracing']
                if save_ray_tracing == 'T':
                    if verbose:
                        print("[!] Saving the ray tracing maps...")
                    kappa_lensing = np.load(path_output+f'/kappa_lensing_sim{sim_idx:05d}_nside{nside:04d}.npy')
                    output_['kappa_lensing'] = kappa_lensing
                    del kappa_lensing
                    output_['gamma_lensing'] = gamma_lensing
                    output_['z_bin_edges'] = z_bin_edges

                #Average upon the redshift bins
                weight_w_redshift = config['redshift_distribution']['weight_w_redshift']
                if weight_w_redshift == 'T':
                    if verbose:
                        print("[!] Computing the shear map on the redshift bins...")
                        print("[!] Load the redshift distribution...")

                    path_dndz = config['redshift_distribution']['path_dndz']
                    
                    redshift_distr = np.loadtxt(path_dndz)

                    nbins = config['redshift_distribution']['nbins']
                    assert redshift_distr.shape[1] == nbins+1, "The redshift distribution file does not correspond to the number of bins."
                    #Check if the m_bias prior is provided
                    if 'm_bias' in config['redshift_distribution']:
                        m_bias_prior = np.loadtxt(config['redshift_distribution']['m_bias'])
                        m_bias_prior = m_bias_prior.reshape((-1, 2))
                        assert m_bias_prior.shape[0] == nbins, "The m_bias file does not correspond to the number of bins."
                    else:
                        m_bias_prior = None

                    #Check if the photo_z systematic prior is provided
                    if 'delta_z' in config['redshift_distribution']:
                        delta_z_prior = np.loadtxt(config['redshift_distribution']['delta_z'])
                        delta_z_prior = delta_z_prior.reshape((-1, 2))
                        assert delta_z_prior.shape[0] == nbins, "The delta_z file does not correspond to the number of bins."
                    else:
                        delta_z_prior = None

                    if validation_plot:
                        validation_plot_dndz(redshift_distr, nbins)
                        

                    save = config['redshift_distribution']['save']

                    if config['shape_noise']['add_shape_noise'] == 'T':
                        #Load the catalog to avoid repeating costly operations on the catalog.
                        if verbose:
                            print("[!] Load the galaxy catalog...")
                        path_cat = config['shape_noise']['path_gal'] 
                        ra, dec, e1, e2, w = load_sources(path_cat, config['shape_noise'])

                        ra, dec, e1, e2 = get_rotation(ra, dec, e1, e2, j, k, verbose)
                    #Compute the shear map weighted by the redshift distribution
                    for i in range(nbins):
                        output_[f'bin_{i+1}'] = {}
                        nuisance_parameters[f'bin_{i+1}'] = {}
                        dndz = redshift_distr[:, i+1]
                        z = redshift_distr[:, 0]
                        if delta_z_prior is not None:
                            delta_z = np.random.normal(loc=delta_z_prior[i, 0], scale=delta_z_prior[i, 1])
                            nuisance_parameters[f'bin_{i+1}']['delta_z'] = delta_z
                            if verbose:
                                print(f"[!] Adding photo-z systematic error of {delta_z} in bin {i+1}...")
                            z_shift = z + delta_z
                            dndz = np.interp(z, z_shift, dndz)
                        if b_sc != 0.0:
                            overdensity_array = np.load(path_output+f'/overdensity_array_sim{sim_idx:05d}_nside{nside:04d}.npy')
                        else:
                            overdensity_array = None
                        gamma_bar, noise_factor = weight_map_w_redshift(gamma_lensing, z_bin_edges, (dndz, z), bias=b_sc, overdensity_array=overdensity_array, verbose=verbose)
                        #noise factor is a scaling factor for the noise map to avoid double counting source clustering effects.
                        del overdensity_array

                        #Add multiplicative shear bias
                        if m_bias_prior is not None:
                            m_bias = np.random.normal(loc=m_bias_prior[i, 0], scale=m_bias_prior[i, 1])
                            nuisance_parameters[f'bin_{i+1}']['m_bias'] = m_bias
                            if verbose:
                                print(f"[!] Adding multiplicative shear bias of {m_bias} in bin {i+1}...")
                            gamma_bar = gamma_bar * (1+m_bias)

                        if config['redshift_distribution']['save_cl'] == 'T':
                            if verbose:
                                print(f"[!] Computing the cls in bin {i+1}...")
                            start_cls = time.time()
                            kappa_lensing = np.load(path_output+f'/kappa_lensing_sim{sim_idx:05d}_nside{nside:04d}.npy')
                            kappa_bar, _ = weight_map_w_redshift(kappa_lensing, z_bin_edges, (dndz, z), verbose=verbose)
                            cls = hp.anafast([kappa_bar, gamma_bar.real, gamma_bar.imag], pol=True, lmax=3*nside, use_pixel_weights=True)
                            output_[f'bin_{i+1}'][f'cl_FS_gamma'] = cls
                            del kappa_lensing, kappa_bar
                            if verbose:
                                print(f"[!] Cls computed in {(time.time()-start_cls)/60:.2f} minutes...")
                        if save == 'T':
                            if verbose:
                                print(f"[!] Saving the weighted maps for redshift bin {i+1}...")
                            output_[f'bin_{i+1}'][f'gamma_weighted'] = gamma_bar

                        #Mask and add shape noise
                        if config['shape_noise']['add_shape_noise'] == 'T':
                            if verbose:
                                print("[!] Adding shape noise and applying mask to the shear map...")
                            #!!!does not take into account the different galaxies in different bins!!!
                            masked_shear_map, noise_map, idx_ = add_shape_noise(gamma_bar, ra, dec, e1, e2, w)
                            noise_map = noise_map * noise_factor
                            del gamma_bar

                            save = config['shape_noise']['save']
                            if save == 'T':
                                if verbose:
                                    print("[!] Saving the masked shear map and the noise map...")
                                output_[f'bin_{i+1}'][f'masked_shear_map'] = masked_shear_map
                                output_[f'bin_{i+1}'][f'noise_map'] = noise_map
                                output_[f'bin_{i+1}'][f'idx'] = idx_
                                del masked_shear_map, noise_map, idx_

                        if config['psf_systematic']['add_systematic']:
                            if verbose:
                                print(f"[!] Adding the PSF systematic error in bin {i+1}...")
                            path_psf = config['psf_systematic']['path_psf']
                            prior_params = np.load(config['psf_systematic']['path_prior_params'], allow_pickle=True).item()
                            alpha, beta, eta, sys_map, idx_star = sample_sys_map(path_psf, nside, config['psf_systematic'], prior_params[f'bin{i+1}'], verbose, i=j, j=k)
                            output_[f'bin_{i+1}'][f'sys_map'] = sys_map
                            output_[f'bin_{i+1}'][f'idx_star'] = idx_star
                            nuisance_parameters[f'bin_{i+1}'][f'alpha'] = alpha
                            nuisance_parameters[f'bin_{i+1}'][f'beta'] = beta
                            nuisance_parameters[f'bin_{i+1}'][f'eta'] = eta
                            del sys_map, idx_star

                    del ra, dec, e1, e2, w

                        

                output_['nuisance_parameters'] = nuisance_parameters
                #Save the output
                np.save(path_output+f'/forward_model_sim{sim_idx:05d}_nside{nside:04d}_rot{j}{k}_noisereal{noise_real}.npy', output_)
                output_.clear()
                del output_
                gc.collect()
                if verbose:
                    print(f"[!] The forward model for rotation {j}{k} is done.")
                    print(f"[!] The forward model for rotation {j}{k} took {(time.time()-start_)/60:.2f} minutes.")

    os.remove(path_output+f'/kappa_lensing_sim{sim_idx:05d}_nside{nside:04d}.npy')
    os.remove(path_output+f'/overdensity_array_sim{sim_idx:05d}_nside{nside:04d}.npy')
    os.remove(path_output+f'/gamma_lensing_sim{sim_idx:05d}_nside{nside:04d}.npy')     
    print(f"[!] The forward model for simulation {sim_idx} is done.")
    print(f"[!] The forward model for simulation {sim_idx} took {(time.time()-start)/60:.2f} minutes.")

