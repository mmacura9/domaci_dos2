# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 23:37:16 2021

@author: mm180261d
"""
import skimage
from pylab import *
import numpy as np
from skimage import io
import math

import matplotlib.pyplot as plt

if __name__ == "__main__":
    img_in = skimage.img_as_float(io.imread('../sekvence/etf_blur.tif'))
    kernel = np.zeros(img_in.shape, dtype = float)
    kernel1 = skimage.img_as_float(io.imread('../sekvence/kernel.tif'))
    kernel[:kernel1.shape[0], :kernel1.shape[1]] = kernel1
    plt.figure()
    io.imshow(img_in)
    plt.figure()
    io.imshow(kernel, cmap = 'gray')
    # FFT slike i kernela
    img_fft = np.fft.fftshift(np.fft.fft2(img_in))
    kernel_fft = np.fft.fftshift(np.fft.fft2(kernel))
    kernel_fft[math.floor(kernel_fft.shape[0]/2), math.floor(kernel_fft.shape[1]/2)] = 1
    
    plt.figure()
    io.imshow(log(1+abs(kernel_fft)), cmap = 'gray')
    
    plt.figure()
    io.imshow(log(1+abs(np.fft.fftshift(np.fft.fft2(kernel1)))), cmap = 'gray')
    
    #Vinerov filtar
    F_est_fft = img_fft/kernel_fft*(abs(kernel_fft)**2/(abs(kernel_fft)**2+1))
    f_est = np.fft.ifft2(np.fft.ifftshift(F_est_fft))
    
    plt.figure()
    io.imshow(real(f_est), cmap= 'gray')
    