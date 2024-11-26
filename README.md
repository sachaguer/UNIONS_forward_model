# UNIONS forward model

This repository contains useful scripts to create weak lensing mock maps where systematics are forward modelled. This pipeline was designed to be used for the UNIONS survey but should be modular enough to be used in other context.

The input of the pipeline is a simulation with provided lightcones between some redshift bin edges. The pipeline was first developed to build weak lensing maps using the Gower Street simulations suite but could be adapted to other simulations suite. The ray-tracing to get a convergence map from the lightcones can be done with `glass` or `bornraytrace`. The shear maps are obtained from the convergence map using Kaiser-Squire formula.

The following systematics are forward modelled:
- Intrinsic alignement (NLA model): $A_{\rm IA}$ and $\eta_{\rm IA}$ are drawn from given flat priors.
- PSF systematics (rho/tau-statistics error model): parameters $\alpha$, $\beta$ and $\eta$ are drawn from a prior informed by rho- and tau-statistics.
- Multiplicative bias: $m$ is drawn from a prior informed by image simulations.
- Redshift distribution uncertainty $\Delta z$.
- The noise is modelled by randomly rotating the galaxies from a given galaxy catalog.

## Running the pipeline

To run the pipeline, a config file has to be given in input of the script `run.py`. The default path to the config file is `./config.yaml` which provides an example of config file for the pipeline.

## Output of the pipeline

The output of the pipeline is one (or several) `.npy` file. They contain the configuration with which the forward model was run, the cosmological and nuisance parameters and the weak lensing maps provided in output in each tomographic bin. The number of noise realisations can be tuned with the parameter `n_noise_real` in the configuration file. The parameters `rot_ra` and `rot_dec` allow to choose the rotation of the footprint to use.