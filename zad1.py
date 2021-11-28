# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:10:10 2021

@author: mm180261d
"""
import skimage
from skimage import io
import matplotlib.pyplot as plt

from pylab import *
import numpy as np


if __name__ == "__main__":
    img_in = skimage.img_as_float(imread('../sekvence/half_tone.jpg'))
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    io.imshow(img_in)
    for i in range(3):
        img_fft = np.fft.fftshift(np.fft.fft2(img_in[:, :, i]))
        img_amp = abs(img_fft)/np.max(abs(img_fft))
        output = "zad1/fft_" + str(i) + ".jpg"
        imsave(output, np.log(1+img_amp), cmap='gray');