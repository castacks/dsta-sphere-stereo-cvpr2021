import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def get_filename_parts(fn):
    p = os.path.split(fn)

    if ( "" == p[0] ):
        p = (".", p[1])

    f = os.path.splitext(p[1])

    return [ p[0], f[0], f[1] ]

def test_directory(d):
    if ( not os.path.isdir(d) ):
        os.makedirs(d)

def test_directory_by_filename(fn):
    parts = get_filename_parts(fn)

    return test_directory(parts[0])

# ==============================================================================

def read_image(fn, dtype=np.uint8):
    if ( not os.path.isfile(fn) ):
        raise Exception('%s does not exist. ' % (fn))

    return cv2.imread(fn, cv2.IMREAD_UNCHANGED).astype(dtype)

def read_compressed_float(fn, typeStr='<f4'):
    if ( not os.path.isfile(fn) ):
        raise Exception('%s does not exist. ' % (fn))

    return np.squeeze( 
        cv2.imread(fn, cv2.IMREAD_UNCHANGED).view(typeStr), 
        axis=-1 )

# ==============================================================================

def write_image(fn, img):
    test_directory_by_filename(fn)
    cv2.imwrite( fn, img )

def write_float_image_normalized(fn, img):
    """
    fn: The output filename.
    img: A single channel image.

    Only supports writing PNG image.
    """

    # Test the output directory.
    parts = get_filename_parts(fn)

    if ( not os.path.isdir(parts[0]) ):
        os.makedirs(parts[0])
    
    # Normalize img.
    img = img.astype(np.float32)
    img = img - img.min()
    img = img / img.max()
    img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)

    # Save the image.
    cv2.imwrite(fn, img)

    return img

def write_float_image_normalized_clip(fn, img, m0, m1):
    """
    fn: The output filename.
    img: A single channel image.
    m0, m1: the minimum and maximum value.

    Only supports writing PNG image.
    """

    if ( m0 >= m1 ):
        raise Exception("Wrong clipping bounds: {}, {}.".format(m0, m1))

    return write_float_image_normalized( fn, np.clip(img, m0, m1) ) 

def write_float_image_fixed_normalization(fn, img, m0, m1):
    """
    fn: The output filename.
    img: A single channel image.
    m0, m1: The lower and upper bounds. Values in img will be
    clipped by m0 an m1 before normalization.

    Only supports writing PNG image.
    """

    # Test the output directory.
    parts = get_filename_parts(fn)

    if ( not os.path.isdir(parts[0]) ):
        os.makedirs(parts[0])

    # Check the dimension of img.
    assert (img.ndim == 2), "img must be a 2D array img.shape = {}. ".format( img.shape )

    # Check the bounds.
    assert (m1 > m0), "m1 ({}) must larger than m0 ({}). ".format( m1, m0 )

    # Conver img to float type.
    img = img.astype(np.float32)

    # Clip the value.
    img = np.clip(img, m0, m1)

    # Normalization.
    img = (img - m0) / ( m1 - m0 )
    img = img * 255
    img = img.astype(np.uint8)

    # Save the image.
    cv2.imwrite(fn, img)

    return img

def write_float_RGB(fn, img):
    """
    fn: The output filename.
    img: A 3-channel image.
    
    This function first clip the input img to 0-255, then convert
    img to uint8 type.

    Only supports writing PNG image.
    """

    # Test the output directory.
    parts = get_filename_parts(fn)

    if ( not os.path.isdir(parts[0]) ):
        os.makedirs(parts[0])

    # Check the dimension of img.
    assert (img.ndim == 3), "img.shape = {}. ".format( img.shape )

    # Clip the floating point number to 0-255 before converting to uint8.
    img = np.clip(img, 0, 255)

    # Conver img to float type.
    img = img.astype(np.uint8)

    # Save the image.
    cv2.imwrite(fn, img)

    return img

def write_float_image_plt(fn, img):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.axis("off")
    im = ax.imshow(img)
    fig.colorbar(im)
    fig.savefig(fn)
    plt.close(fig)

def write_float_image_plt_clip(fn, img, m0, m1):
    write_float_image_plt( fn, np.clip( img, m0, m1 ) )
    
def write_compressed_float(fn, img, typeStr='<u1'):
    assert(img.ndim == 2), 'img.ndim = {}'.format(img.ndim)
    
    # Check if the input array is contiguous.
    if ( not img.flags['C_CONTIGUOUS'] ):
        img = np.ascontiguousarray(img)
    
    test_directory_by_filename(fn)

    dummy = np.expand_dims( img, 2 )
    cv2.imwrite( fn, dummy.view(typeStr) )