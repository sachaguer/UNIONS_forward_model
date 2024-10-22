import os
import yaml
import argparse

import numpy as np
import scipy.stats as stats
import healpy as hp
import matplotlib.pyplot as plt
import camb
from astropy.io import fits

from forward_model import forward, weight_map_w_redshift, add_shape_noise
from psf_systematic import sample_sys_map

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # You can change this font if you prefer
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
    kappa_lensing_bar = weight_map_w_redshift(kappa_lensing, z_bin_edges, (dndz, z), verbose=verbose)

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

parser = argparse.ArgumentParser()

parser.add_argument("-c", "--config", help="Path to the configuration file.", type=str, default='config.yaml')

if __name__ == '__main__':
    # Load the configuration file
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    verbose = config['verbose']
    validation_plot = config['validation_plot']

    path_output = config['simulation']['path_output']
    output = {}
    nuisance_parameters = {}

    #Check if output directory exists
    if not os.path.exists(path_output):
        if verbose:
            print(f"[!] Creating the output directory {path_output}.")
        os.makedirs(path_output)

    if validation_plot and not os.path.exists(path_output+'/Plots/'):
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

    if verbose:
        print("[!] Performing the forward model...")
        print(f"[!] The chosen resolution is nside={nside} which corresponds to {hp.nside2resol(nside, arcmin=True):.2f} arcmin.")
    
    if add_ia:
        A_ia = np.random.uniform(low=config['intrinsic_alignment']['prior_A_ia'][0], high=config['intrinsic_alignment']['prior_A_ia'][1])
        eta_ia = np.random.uniform(low=config['intrinsic_alignment']['prior_eta_ia'][0], high=config['intrinsic_alignment']['prior_eta_ia'][1])
        if verbose:
            print("[!] Adding the intrinsic alignment to the shear maps...")
        nuisance_parameters['A_ia'] = A_ia
        nuisance_parameters['eta_ia'] = eta_ia

    kappa_lensing, kappa_ia, gamma_lensing, z_bin_edges, cosmo_params = forward(
        path_sims, path_info, sim_name=sim_name, sim_number=sim_idx, nside=nside, nside_intermediate=nside_intermediate, method=ray_tracing_method, add_ia=add_ia, A_ia=A_ia, eta_ia=eta_ia, verbose=verbose
    )
    output['cosmo_params'] = cosmo_params

    #Saves the output if required
    save_ray_tracing = config['ray_tracing']['save_ray_tracing']
    if save_ray_tracing:
        if verbose:
            print("[!] Saving the ray tracing maps...")
        output[f'kappa_lensing'] = kappa_lensing
        if add_ia:
            output['kappa_ia'] = kappa_ia
        output['gamma_lensing'] = gamma_lensing
        output['z_bin_edges'] = z_bin_edges

    #Average upon the redshift bins
    weight_w_redshift = config['redshift_distribution']['weight_w_redshift']
    if weight_w_redshift:
        if verbose:
            print("[!] Computing the shear map on the redshift bins...")
            print("[!] Load the redshift distribution...")

        path_dndz = config['redshift_distribution']['path_dndz']
        
        redshift_distr = np.loadtxt(path_dndz)

        nbins = config['redshift_distribution']['nbins']
        assert redshift_distr.shape[1] == nbins+1, "The redshift distribution file does not correspond to the number of bins."

        if validation_plot:
            validation_plot_dndz(redshift_distr, nbins)
            

        save = config['redshift_distribution']['save']
        #Compute the shear map weighted by the redshift distribution
        for i in range(nbins):
            dndz = redshift_distr[:, i+1]
            z = redshift_distr[:, 0]
            gamma_bar = weight_map_w_redshift(gamma_lensing, z_bin_edges, (dndz, z), verbose=verbose)

            if save:
                if verbose:
                    print(f"[!] Saving the weighted maps for redshift bin {i+1}...")
                output[f'gamma_weighted_bin{i+1}'] = gamma_bar

            #Mask and add shape noise
            if config['shape_noise']['add_shape_noise']:
                if verbose:
                    print("[!] Adding shape noise and applying mask to the shear map...")
                    print("[!] Load the galaxy catalog...")
                path_cat = config['shape_noise']['path_gal']
                cat_gal = fits.getdata(path_cat)
                ra = cat_gal[config['shape_noise']['ra_col']]
                dec = cat_gal[config['shape_noise']['dec_col']]
                e1 = cat_gal[config['shape_noise']['e1_col']]
                e2 = cat_gal[config['shape_noise']['e2_col']]
                w = cat_gal[config['shape_noise']['w_col']]

                masked_shear_map, noise_map = add_shape_noise(gamma_bar, ra, dec, e1, e2, w)

                save = config['shape_noise']['save']
                if save:
                    if verbose:
                        print("[!] Saving the masked shear map and the noise map...")
                    output[f'masked_shear_map_bin{i+1}'] = masked_shear_map
                    output[f'noise_map_bin{i+1}'] = noise_map

            if config['psf_systematic']['add_systematic']:#!!! To implement the prior for alpha, beta, eta for each bin.!!!
                if verbose:
                    print(f"[!] Adding the PSF systematic error in bin {i+1}...")
                path_psf = config['psf_systematic']['path_psf']
                prior_params = np.load(config['psf_systematic']['path_prior_params'], allow_pickle=True).item()
                alpha, beta, eta, sys_map = sample_sys_map(path_psf, nside, config['psf_systematic'], prior_params[f'bin{i+1}'], verbose)
                output[f'sys_map_bin{i+1}'] = sys_map
                nuisance_parameters[f'alpha_bin{i+1}'] = alpha
                nuisance_parameters[f'beta_bin{i+1}'] = beta
                nuisance_parameters[f'eta_bin{i+1}'] = eta

            

    output['nuisance_parameters'] = nuisance_parameters
    output['config'] = config
    #Save the output
    np.save(path_output+f'/forward_model_sim{sim_idx:05d}_nside{nside:04d}.npy', output)
    if verbose:
        print("[!] The forward model is done.")

