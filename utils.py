import numpy as np
import healpy as hp

def downgrade_lightcone(lightcone, nside_out):
    """
    Downgrade a lightcone to a lower resolution (nside_out < nside_in)
    """
    nside_in = hp.npix2nside(len(lightcone))
    if nside_in < nside_out:
        raise ValueError("nside_out must be smaller than nside_in")
    if nside_in == nside_out:
        return lightcone
    lightcone_down = hp.ud_grade(lightcone, nside_out)
    print(f"[!] Done.")
    return lightcone_down
