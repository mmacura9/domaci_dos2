# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 21:10:18 2021

@author: mm180261d
"""

import skimage
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from scipy import ndimage

from pylab import *
import numpy as np

def noise_surr_var(img_in: np.array, r: int) -> []:
    box = np.ones([2*r+1, 2*r+1], dtype=float)/(2*r+1)**2
    
    box_filtar = ndimage.correlate(img_in, box)
    square_box_filtar = ndimage.correlate(img_in**2, box)
    
    sigma_xy = square_box_filtar - box_filtar**2
    
    histogram, bins = np.histogram(sigma_xy.flatten(), bins=256)
    arg = np.argmax(histogram)
    
    sigma_n = bins[arg]
    return [sigma_n, sigma_xy]

if __name__ == "__main__":
    img_in = skimage.img_as_float(io.imread('..\sekvence\lena_noise.tif'))
    io.imshow(img_in)
    r = 4
    box = np.ones([2*r+1, 2*r+1], dtype=float)/(2*r+1)**2
    
    box_filtar = ndimage.correlate(img_in, box)
    square_box_filtar = ndimage.correlate(img_in**2, box)
    
    sigma_xy = square_box_filtar - box_filtar**2
    
    histogram, bins = np.histogram(sigma_xy.flatten(), bins=256)
    arg = np.argmax(histogram)
    
    sigma_n = bins[arg]
    
    w = sigma_n / sigma_xy
    w[w>1] = 1
    
    img_out = img_in - w * (img_in - box_filtar)
    
    plt.figure()
    io.imshow(img_out)
    