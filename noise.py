
import numpy as np
import os
import cv2


NOISE_TYPE_GAUSS = 'gauss'
NOISE_TYPE_SALT_PEPPER = 's&p'
NOISE_TYPE_POISSON = 'poisson'
NOISE_TYPE_SPECKLE = 'speckle'

def noisify(noise_type, image):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    noise_type : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """
    if noise_type == NOISE_TYPE_GAUSS:
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_type == NOISE_TYPE_SALT_PEPPER:
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == NOISE_TYPE_POISSON:
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == NOISE_TYPE_SPECKLE:
        gauss = None
        if len(image.shape) == 2:
            row,col = image.shape #row,col & channel if more than 1 color channel present
            gauss = np.random.randn(row,col)
            gauss = gauss.reshape(row,col)        
        elif len(image.shape) == 2:
            row,col,ch = image.shape #row,col & channel if more than 1 color channel present
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy