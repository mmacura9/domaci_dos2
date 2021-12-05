# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:10:10 2021

@author: mm180261d
"""
from skimage import color
import skimage
from skimage import io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pylab import *
import numpy as np
from scipy.signal import find_peaks
from scipy import ndimage

def bandpass_filtar(P: int, Q: int) -> np.array:
    bandpass = np.zeros([P, Q], dtype=float)
    D0 = math.sqrt(700)
    D1 = 400
    for u in range(P):
        for v in range(Q):
            D = math.sqrt((u-P/2)**2 + (v-Q/2)**2)
            if D**2 > D0**2 and D**2 < D1**2:
                bandpass[u, v] = exp(-D**2/700)
            else:
                bandpass[u, v] = 1
    return bandpass

def gauss_filtar(P: int, Q: int) -> np.array:
    gauss = np.zeros([P, Q])
    sigma = 1000
    for u in range(P):
        for v in range(Q):
            D = (u - P/2)**2 + (v - Q/2)**2
            gauss[u, v] = exp(-D/sigma)
    return gauss

def batervort_filtar(P: int, Q: int) -> np.array:
    batervort = np.zeros([P, Q])
    D0 = 15
    for u in range(Q):
        for v in range(P):
            D = math.sqrt((u - Q/2)**2 + (v - P/2)**2)
            batervort[u, v] = 1/(1+(D/D0)**4)
    return batervort

def line_filtar(P: int, Q: int) -> np.array:
    filtar = np.ones([P, Q], dtype=float)
    filtar[math.floor(P/2)-1: math.floor(P/2)+2, :] = 0
    filtar[:, math.floor(Q/2)-1: math.floor(Q/2)+2] = 0
    filtar[math.floor(P/2)-10: math.floor(P/2)+11, math.floor(Q/2)-10: math.floor(Q/2)+11] = 1
    return filtar

# def notch_filtar(P: int, Q: int, img_fft: np.array) -> np.array:
#     peaks, _ = find_peaks(abs(img_fft).flatten(), height=(10, np.max(log(abs(img_fft)))-0.1))
#     filtar = np.ones([P, Q], dtype=float)
#     for peak in peaks:
#         gauss = 1 - gauss_filtar(P, Q, [math.floor(peak/img_fft.shape[1]), peak%img_fft.shape[1]])
#         filtar = filtar * gauss
#     return filtar

if __name__ == "__main__":
    img = skimage.img_as_float(imread('../sekvence/half_tone.jpg'))
    img_in = np.zeros([img.shape[0]*2, img.shape[1]*2,3], dtype = float)
    img_in[:img.shape[0], :img.shape[1]] = img
    plt.figure(num=None, figsize=(8, 6), dpi=80)
    io.imshow(img_in)
    
    line = line_filtar(img_in.shape[0], img_in.shape[1])
    
    img_out_fft = np.zeros(img_in.shape, dtype = complex)
    img_out = np.zeros(img_in.shape)
    gauss = gauss_filtar(img_in.shape[0], img_in.shape[1])
    for i in range(3):
        img_fft = np.fft.fftshift(np.fft.fft2(img_in[:, :, i]))
        gauss = bandpass_filtar(img_in.shape[0], img_in.shape[1])
        img_amp = abs(img_fft)
        # x = list(range(img_fft.shape[0]))
        # y = list(range(img_fft.shape[1]))
        # X, Y = meshgrid(x, y)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.plot_surface(X, Y, 1 + img_amp)
        imsave("zad1/fft_"+str(i)+".jpg", log(1 + img_amp), cmap='gray')
        img_out_fft[:, :, i] = img_fft*line
        img_out_fft[:, :, i] = img_out_fft[:, :, i] * gauss
        img_amp = abs(img_out_fft[:, :, i])
        img_out[:, :, i] = real(np.fft.ifft2(np.fft.ifftshift(img_out_fft[:, :, i])))
        output = "zad1/filtrated_fft_" + str(i) + ".jpg"
        imsave(output, np.log(1+img_amp), cmap='gray')
    img_out[img_out > 1] = 1
    img_out[img_out < 0] = 0
    img_out = img_out[:img.shape[0], :img.shape[1]]
    
    # laplacian = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    # img_out_yuv = color.rgb2yuv(img_out)
    # compY = img_out_yuv[:,:,0]
    # filtratedCompY = ndimage.correlate(compY, laplacian)
    # img_out_yuv[:, :, 0] = compY - filtratedCompY
    # img_out_yuv[img_out_yuv[:,:,0]>1,0] = 1
    # img_out_yuv[img_out_yuv[:,:,0]<0,0] = 0
    
    # img_out = color.yuv2rgb(img_out_yuv)
    # img_out[img_out>1] = 1
    # img_out[img_out<0] = 0
    plt.figure()
    io.imshow(img_out)
    imsave("zad1/izlaz.jpg", img_out)