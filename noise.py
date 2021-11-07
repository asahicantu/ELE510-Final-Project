
import numpy as np
import os
import cv2
import findpeaks

# filters parameters
# window size
winsize = 15
# damping factor for frost
k_value1 = 2.0
# damping factor for lee enhanced
k_value2 = 1.0
# coefficient of variation of noise
cu_value = 0.25
# coefficient of variationfor lee enhanced of noise
cu_lee_enhanced = 0.523
# max coefficient of variation for lee enhanced 
cmax_value = 1.73


NOISE_TYPE_GAUSS = 'gauss'
NOISE_TYPE_SALT_PEPPER = 's&p'
NOISE_TYPE_POISSON = 'poisson'
NOISE_TYPE_SPECKLE = 'speckle'

'''None Local Means denosify algorithm'''
DENOISE_TYPE_NLM = 'nlm'
DENOISE_TYPE_NONE = None
DENOISE_TYPE_LEE = 'lee'
DENOISE_TYPE_LEE_ENHANCED = 'lee_enhanced'
DENOISE_TYPE_KUAN = 'kuan'
DENOISE_TYPE_FASTNL = 'fastnl'
DENOISE_TYPE_BILATERAL = 'bilateral'
DENOISE_TYPE_FROST = 'frost'
DENOISE_TYPE_MEDIAN = 'median'
DENOISE_TYPE_MEAN = 'mean'

DENOISE_FILTERS = [None, 'lee', 'lee_enhanced', 'kuan', 'fastnl', 'bilateral', 'frost', 'median', 'mean']

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
        elif len(image.shape) == 3:
            row,col,ch = image.shape #row,col & channel if more than 1 color channel present
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy





def denosify(denoise_type, img):
    img = (img * 255).astype(np.uint8) ## avoid unsupported cv2 type exception
    denoise = None
    if denoise_type == DENOISE_TYPE_NLM:
        if len(img.shape) == 2:
            denoise = cv2.fastNlMeansDenoising(img,None,10,7,21)
        elif len(img.shape) == 3:
            denoise = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    if denoise_type == DENOISE_TYPE_FASTNL:
        denoise = findpeaks.stats.denoise(img, method = DENOISE_TYPE_FASTNL, window = winsize)
    if denoise_type == DENOISE_TYPE_BILATERAL:
        denoise = findpeaks.stats.denoise(img, method = DENOISE_TYPE_BILATERAL, window = winsize)
    if denoise_type == DENOISE_TYPE_FROST:
        denoise = findpeaks.frost_filter(img, damping_factor = k_value1, win_size = winsize)
    if denoise_type == DENOISE_TYPE_KUAN:
        denoise = findpeaks.kuan_filter(img, win_size = winsize, cu = cu_value)
    if denoise_type == DENOISE_TYPE_LEE:
        denoise = findpeaks.lee_filter(img, win_size = winsize, cu = cu_value)
    if denoise_type == DENOISE_TYPE_LEE_ENHANCED:
        denoise = findpeaks.lee_enhanced_filter(img, win_size = winsize, k = k_value2, cu = cu_lee_enhanced, cmax = cmax_value)
    if denoise_type == DENOISE_TYPE_MEAN:
        denoise = findpeaks.mean_filter(img, win_size = winsize)
    if denoise_type == DENOISE_TYPE_MEDIAN:
        denoise = findpeaks.median_filter(img, win_size = winsize) 
    return denoise