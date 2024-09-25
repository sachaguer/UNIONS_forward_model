import os
import yaml

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import camb
from astropy.io import fits

from forward_model import forward, weight_map_w_redshift, add_shape_noise

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

if __name__ == '__main__':
    # Load the configuration file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    verbose = config['verbose']
    validation_plot = config['validation_plot']

    path_output = config['simulation']['path_output']

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
        print(f"[!] The chose resolution is nside={nside} which corresponds to {hp.nside2resol(nside, arcmin=True):.2f} arcmin.")
    
    if add_ia:
        A_ia = config['intrinsic_alignment']['A_ia']
        eta_ia = config['intrinsic_alignment']['eta_ia']

    kappa_lensing, kappa_ia, gamma_lensing, z_bin_edges, cosmo_params = forward(
        path_sims, path_info, sim_name=sim_name, sim_number=sim_idx, nside=nside, nside_intermediate=nside_intermediate, method=ray_tracing_method, add_ia=add_ia, A_ia=A_ia, eta_ia=eta_ia, verbose=verbose
    )

    save_ray_tracing = config['ray_tracing']['save_ray_tracing']

    if save_ray_tracing:
        if verbose:
            print("[!] Saving the ray tracing maps...")
        np.save(path_output+f'/kappa_lensing_sim{sim_idx:05d}.npy', kappa_lensing)
        if add_ia:
            np.save(path_output+f'/kappa_ia_sim{sim_idx:05d}.npy', kappa_ia)
        np.save(path_output+f'/gamma_lensing_sim{sim_idx:05d}.npy', gamma_lensing)
        np.save(path_output+f'/z_bin_edges_sim{sim_idx:05d}.npy', z_bin_edges)

    weight_w_redshift = config['redshift_distribution']['weight_w_redshift']

    if weight_w_redshift:
        if verbose:
            print("[!] Computing the shear map on the redshift bins...")
            print("[!] Load the redshift distribution...")

        path_dndz = config['redshift_distribution']['path_dndz']
        z, dndz = np.loadtxt(path_dndz, unpack=True)

        if validation_plot:
            plt.figure()

            plt.plot(z, dndz, label='Redshift distribution')
            plt.xlabel(r'$z$')
            plt.ylabel(r'$n(z)$')
            plt.legend()
            plt.savefig(path_output+'/Plots/dndz.png')
            plt.close()
        
            kappa_lensing_bar = weight_map_w_redshift(kappa_lensing, z_bin_edges, (dndz, z), verbose=verbose)

            hp.mollview(kappa_lensing_bar, title='Convergence map weighted by the redshift distribution', cmap='inferno', max=0.02)
            plt.savefig(path_output+'/Plots/kappa_lensing_bar.png')
            plt.close()

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

        gamma_bar = weight_map_w_redshift(gamma_lensing, z_bin_edges, (dndz, z), verbose=verbose)

        if validation_plot:
            hp.mollview(gamma_bar.real, title=r'$\gamma_1$ map weighted by the redshift distribution', cmap='inferno')
            plt.savefig(path_output+'/Plots/gamma1_fullsky.png')
            plt.close()

            hp.mollview(gamma_bar.imag, title=r'$\gamma_2$ map weighted by the redshift distribution', cmap='inferno')
            plt.savefig(path_output+'/Plots/gamma2_fullsky.png')
            plt.close()

        save = config['redshift_distribution']['save']

        if save:
            if verbose:
                print("[!] Saving the weighted maps...")
            np.save(path_output+f'/gamma_lensing_bar_sim{sim_idx:05d}.npy', gamma_bar)

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

            if validation_plot:
                hp.mollview(masked_shear_map.real, title=r'$\gamma_1$ map with shape noise', cmap='seismic')
                plt.savefig(path_output+'/Plots/gamma1_masked_map.png')
                plt.close()

                hp.mollview(masked_shear_map.imag, title=r'$\gamma_2$ map with shape noise', cmap='seismic')
                plt.savefig(path_output+'/Plots/gamma2_masked_map.png')
                plt.close()

                hp.mollview(noise_map.real, title=r'$\gamma_1$ shape noise map', cmap='seismic')
                plt.savefig(path_output+'/Plots/gamma1_noise_map.png')
                plt.close()

                hp.mollview(noise_map.imag, title=r'$\gamma_2$ shape noise map', cmap='seismic')
                plt.savefig(path_output+'/Plots/gamma2_noise_map.png')
                plt.close()

            save = config['shape_noise']['save']
            if save:
                if verbose:
                    print("[!] Saving the masked shear map and the noise map...")
                np.save(path_output+f'/masked_shear_map_sim{sim_idx:05d}.npy', masked_shear_map)
                np.save(path_output+f'/noise_map_sim{sim_idx:05d}.npy', noise_map)

    if verbose:
        print("[!] The forward model is done.")

