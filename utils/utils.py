import numpy as np


def simple_equalization_8bit(im, percentiles=5):
    """
    Simple 8-bit requantization by linear stretching.

    Args:
        im (np.array): image to requantize
        percentiles (int): percentage of the darkest and brightest pixels to saturate

    Returns:
        numpy array with the quantized uint8 image
    """
    
    mi, ma = np.percentile(im[np.isfinite(im)], (percentiles, 100 - percentiles))
    im = np.clip(im, mi, ma)
    im = (im - mi) / (ma - mi) * 255   # scale
    return im.astype(np.uint8)


def register(im1, im2, di, dj):
    """
    Register images with known displacement.

    Parameters
    ----------
    im1, im2: ndarrays of same shape
        images to register

    di, dj: ints
        displacement along the i- and j- directions to apply to the 2nd image to register it on the first
    """
    assert im1.shape == im2.shape
    m, n = im1.shape
    # make di and dj as small as possible in absolute value, using the periodicity of the translation
    if di > m // 2:
        di = di - m
    if dj > n // 2:
        dj = dj - n
    # compute translated and cropped images
    if di > 0:  # im2 must be translated to the right, i.e. im1 to the left
        im1 = im1[di:]
        im2 = im2[:-di]
    elif di < 0:  # im2 must be translated to the left
        im1 = im1[:di]
        im2 = im2[-di:]
    if dj > 0:  # im2 must be translated downwards, i.e. im1 upwards
        im1 = im1[:, dj:]
        im2 = im2[:, :-dj]
    elif dj < 0:  # im2 must be translated upwards
        im1 = im1[:, :dj]
        im2 = im2[:, -dj:]
    # return result
    return im1, im2
