
import numpy as np
import healpy as hp



def nested2normal(n, nside):
    """If n is an index for nested ordering (healpix), then patch(n, nside) is the corresponding index for normal ordering"""
    if (nside <= 2):
        return n
    r = nside//2
    n = nested2normal(n, r)
    n = n + r * ( 1 - (n%(2*r*r))//(r*r) ) * ( ((n%(r*r))//r) ) - r * ( (n%(2*r*r))//(r*r) ) * ( (r - 1) - (n%(r*r))//r )
    return n

def concatenate1(nside, images, blank_pixels=0):
    blanck = np.full((nside, nside), blank_pixels)

    row1 = np.concatenate((blanck, images[3], blanck), axis=1)
    row2 = np.concatenate((np.swapaxes(images[7, :, ::-1],0,1), np.swapaxes(images[2, :, ::-1], 0, 1), blanck), axis=1)
    row3 = np.concatenate((blanck, images[6, ::-1, :], np.swapaxes(images[1, :,::-1], 0, 1)), axis=1)
    result = np.concatenate((row1, row2, row3), axis=0)
    return result[440:1421, 422:1464]

def concatenate2(nside, images, blank_pixels=0):
    blanck = np.full((nside, nside), blank_pixels)
    
    row1 = np.concatenate((blanck, images[5]), axis=1)
    row2 = np.concatenate((images[4], images[0]), axis=1)
    result = np.concatenate((row1, row2), axis=0)[::-1, ::-1]
    return result[464:660, 199:557]

def get_cells_from_map(nside, map, map_ordering='RING', n_ref=1):
    """Returns 12 images of size nside*nside, corresponding to the nside=n_ref healpix pixels"""

    cell_side = nside//n_ref
    # Reorder map in Nested scheme
    if(map_ordering == 'RING'):
        map_nested = hp.reorder(map, inp="RING", out="NEST")
    elif(map_ordering == 'NEST'):
        map_nested = map.copy()
    else: raise ValueError("map_ordering must be either RING or NEST")

    # Create 12*n_ref*n_ref images
    im_nested = map_nested.reshape((12*n_ref*n_ref, cell_side*cell_side))

    # Reorder images in normal ordering
    im = np.zeros_like(im_nested)
    im[:, nested2normal(np.arange(cell_side*cell_side), cell_side)] = im_nested
    
    return im.reshape((12*n_ref*n_ref, cell_side, cell_side))

def get_images_from_map(nside, map, blank_pixels=0):
    """returns 2 images from a shear map"""
    images = get_cells_from_map(nside, map, n_ref=1)
    return concatenate1(nside, images, blank_pixels), concatenate2(nside, images, blank_pixels)
    