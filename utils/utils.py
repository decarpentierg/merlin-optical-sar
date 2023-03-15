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
