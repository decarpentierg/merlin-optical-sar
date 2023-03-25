import glob
import numpy as np
from scipy import signal


def symmetrization_patch_gen(ima):
    """Apply a translation to the FFT of the main SAR image so as to center its spectrum?

    Arguments
    ---------
    ima: ndarray of shape (m, n, n_channels)
        image to process
    
    Returns
    -------
    ima2: ndarray of shape (m, n, n_channels)
        processed image. Only the 2 first channels are processed.
    """
    sup = ima[:, :, 2:]
    ima = ima[:, :, :2]

    # Take FFT of image
    S = np.fft.fftshift(np.fft.fft2(ima[:, :, 0] + 1j * ima[:, :, 1]))

    # Azimut (ncol)
    p = np.mean(np.abs(S), axis=1)
    sp = p[::-1]
    # correlation
    c = np.real(np.fft.ifft(np.fft.fft(p) * np.conjugate(np.fft.fft(sp))))
    d1 = np.unravel_index(c.argmax(), p.shape[0])
    d1 = d1[0]
    shift_az_1 = int(round(-(d1 - 1) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_1 = np.roll(p, shift_az_1)
    shift_az_2 = int(round(-(d1 - 1 - p.shape[0]) / 2)) % p.shape[0] + int(p.shape[0] / 2)
    p2_2 = np.roll(p, shift_az_2)
    window = signal.gaussian(p.shape[0], std=0.2 * p.shape[0])
    test_1 = np.sum(window * p2_1)
    test_2 = np.sum(window * p2_2)
    # make sure the spectrum is symetrized and zeo-Doppler centered
    if test_1 >= test_2:
        shift_az = shift_az_1
    else:
        shift_az = shift_az_2
    S2 = np.roll(S, shift_az, axis=0)

    # Range (nlin)
    q = np.mean(np.abs(S), axis=0)
    sq = q[::-1]
    # correlation
    cq = np.real(np.fft.ifft(np.fft.fft(q) * np.conjugate(np.fft.fft(sq))))
    d2 = np.unravel_index(cq.argmax(), q.shape[0])
    d2 = d2[0]
    shift_range_1 = int(round(-(d2 - 1) / 2)) % q.shape[0] + int(q.shape[0] / 2)
    q2_1 = np.roll(q, shift_range_1)
    shift_range_2 = int(round(-(d2 - 1 - q.shape[0]) / 2)) % q.shape[0] + int(
        q.shape[0] / 2
    )
    q2_2 = np.roll(q, shift_range_2)
    window_r = signal.gaussian(q.shape[0], std=0.2 * q.shape[0])
    test_1 = np.sum(window_r * q2_1)
    test_2 = np.sum(window_r * q2_2)
    if test_1 >= test_2:
        shift_range = shift_range_1
    else:
        shift_range = shift_range_2

    Sf = np.roll(S2, shift_range, axis=1)
    ima2 = np.fft.ifft2(np.fft.ifftshift(Sf))
    return np.concatenate((np.stack((np.real(ima2), np.imag(ima2)), axis=2), sup), axis=2)

def generate_patches(
    data_dir,
    index,
    pat_size=256,
    step=0,
    stride=64,
    bat_size=4,
    data_aug_times=1,
    method="SAR",
):
    """Generate patches from directory with .npy files.

    Parameters
    ----------
    data_dir: 
    """
    # Define number of channels
    n_channels = 2
    if method == "SAR+SAR" or method == "SAR+OPT":
        n_channels += 1
    if method == "SAR+OPT+SAR":
        n_channels += 2
    
    count = 0

    filepaths = glob.glob(f"{data_dir}/*{index}.npy")
    print(f"number of training data {len(filepaths)}")

    # calculate the number of patches
    for i in range(len(filepaths)):
        img = np.load(filepaths[i])
        im_h = np.size(img, 0)
        im_w = np.size(img, 1)
        for x in range(0 + step, (im_h - pat_size), stride):
            for y in range(0 + step, (im_w - pat_size), stride):
                count += 1
    
    origin_patch_num = count * data_aug_times

    if origin_patch_num % bat_size != 0:
        numPatches = (origin_patch_num / bat_size + 1) * bat_size
    else:
        numPatches = origin_patch_num
    print(
        "total patches = %d , batch size = %d, total batches = %d"
        % (numPatches, bat_size, numPatches / bat_size)
    )

    # data matrix 4-D
    numPatches = int(numPatches)
    inputs = np.zeros((numPatches, pat_size, pat_size, n_channels), dtype="float32")

    count = 0
    # generate patches
    for i in range(len(filepaths)):  # scan through images
        img = np.load(filepaths[i])
        img_s = img
        im_h = np.size(img, 0)
        im_w = np.size(img, 1)
        for x in range(0 + step, im_h - pat_size, stride):
            for y in range(0 + step, im_w - pat_size, stride):
                inputs[count, :, :, :] = symmetrization_patch_gen(
                    img_s[x : x + pat_size, y : y + pat_size, :n_channels]
                )
                count += 1

    # pad the batch
    if count < numPatches:
        to_pad = numPatches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

    return inputs
