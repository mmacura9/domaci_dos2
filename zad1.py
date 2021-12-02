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

def noch_filtar(P: int, Q: int) -> np.array:
    noch = np.zeros([P, Q])
    D0 = math.sqrt(700)
    D1 = math.sqrt(5000)
    for u in range(P):
        for v in range(Q):
            D = math.sqrt((u-P/2)**2 + (v-Q/2)**2)
            if D**2 > D0**2 and D**2 < D1**2:
                noch[u, v] = 0
            else:
                noch[u, v] = 1
    return noch

def gauss_filtar(Q: int, P: int) -> np.array:
    gauss = np.zeros([P, Q])
    D0 = 250
    for u in range(Q):
        for v in range(P):
            D = (u - Q/2)**2 + (v - P/2)**2
            gauss[u, v] = exp(-D/D0)
    return gauss

def batervort_filtar(P: int, Q: int) -> np.array:
    batervort = np.zeros([P, Q])
    D0 = 15
    for u in range(Q):
        for v in range(P):
            D = math.sqrt((u - Q/2)**2 + (v - P/2)**2)
            batervort[u, v] = 1/(1+(D/D0)**4)
    return batervort

if __name__ == "__main__":
    img_in = skimage.img_as_float(imread('../sekvence/half_tone.jpg'))
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    io.imshow(img_in)
    gauss = noch_filtar(img_in.shape[0], img_in.shape[1])
    img_out_fft = np.zeros(img_in.shape, dtype = complex)
    img_out = np.zeros(img_in.shape)
    for i in range(3):
        img_fft = np.fft.fftshift(np.fft.fft2(img_in[:, :, i]))
        # img_amp = abs(img_fft)
        img_out_fft[:, :, i] = img_fft*gauss
        img_amp = abs(img_out_fft[:, :, i])
        img_out[:, :, i] = np.fft.ifft2(np.fft.ifftshift(img_out_fft[:, :, i]))
        output = "zad1/filtrated_fft_" + str(i) + ".jpg"
        imsave(output, np.log(1+img_amp), cmap='gray');
    plt.figure()
    io.imshow(img_out)