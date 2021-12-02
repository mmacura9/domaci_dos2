# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 12:19:22 2021

@author: mm180261d
"""
import matplotlib.pyplot as plt
from pylab import *

import random

import numpy as np
import math
from skimage import io
from skimage import color

def find_max(matrix: np.array) -> int or float:
    return np.max(matrix)
    output = matrix[0][0]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j]>output:
                output = matrix[i][j]
    return output

def find_min(matrix: np.array) -> int or float:
    return np.min(matrix)
    output = matrix[0][0]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j]<output:
                output = matrix[i][j]
    return output

def find_median(matrix: np.array, window: int) -> int or float:
    arr = matrix.flatten()
    middle = math.floor(window*window/2)
    # for i in range(arr.shape[0]):
    #     for j in range(i, arr.shape[0]):
    #         if(arr[j]<arr[i]):
    #             x = arr[i]
    #             arr[i] = arr[j]
    #             arr[j] = x
    #     if i == middle:
    #         return arr[middle]
    arr = np.sort(arr)
    return arr[middle]

def make_mat(img_in: np.array, s_max: int, i: int, j: int) -> np.array:
    window = math.floor(s_max/2)
    if i+window+1 <= img_in.shape[0] and j+window+1 <= img_in.shape[1] and i-window>=0 and j-window>=0:
        return img_in[i-window:i+window+1, j-window:j+window+1]
    
    if i+window+1 > img_in.shape[0] and j+window+1 <= img_in.shape[1] and j-window>=0:
        mat = img_in[i-window:, j-window:j+window+1]
        con = img_in[2*img_in.shape[0]-i-window-1:, j-window:j+window+1]
        return np.concatenate((mat, con), axis=0)
    
    if i-window >= 0 and i+window+1 <= img_in.shape[0] and j+window+1 > img_in.shape[1]:
        mat = img_in[i-window:i+window+1, j-window:]
        con = img_in[i-window:i+window+1, 2*img_in.shape[1]-j-window-1:]
        return np.concatenate((mat, con), axis=1)
    
    if i+window+1>img_in.shape[0] and j+window+1>img_in.shape[1]:
        mat = img_in[i-window:, j-window:]
        con1 = img_in[i-window:, 2*img_in.shape[1]-j-window-1:]
        mat = np.concatenate((mat, con1), axis=1)
        con2 = mat[img_in.shape[0]-i-window-1:, :]
        return np.concatenate((mat, con2), axis=0)
    
    if i-window<0 and j-window>=0 and j+window+1<=img_in.shape[1]:
        mat = img_in[:i+window+1, j-window:j+window+1]
        con = img_in[:window-i, j-window:j+window+1]
        return np.concatenate((mat, con), axis=0)
    
    if i-window>=0 and j-window<0 and i+window<=img_in.shape[0]:
        mat = img_in[i-window:i+window+1, :j+window+1]
        con = img_in[i-window:i+window+1, :window-j]
        return np.concatenate((mat, con), axis=1)
    
    if i-window<0 and j+window+1>img_in.shape[1]:
        mat = img_in[:i+window+1, j-window:]
        con1 = img_in[:i+window+1, 2*img_in.shape[1]-j-window-1:]
        mat = np.concatenate((mat, con1), axis=1)
        con2 = mat[:window-i, :]
        return np.concatenate((mat, con2), axis=0)
    
    if j-window<0 and i+window+1>img_in.shape[0]:
        mat = img_in[i-window:, :j+window+1]
        con1 = img_in[i-window:, :window-j]
        mat = np.concatenate((mat, con1), axis=1)
        con2 = mat[img_in.shape[0]-i-window-1:, :]
        return np.concatenate((mat, con2), axis=0)
    
    mat = img_in[:i+window+1, :j+window+1]
    con1 = img_in[:i+window+1, :window-j]
    mat = np.concatenate((mat, con1), axis=1)
    con2 = mat[:window-i, :]
    return np.concatenate((mat, con2), axis=0)

def dos_median(img_in: np.array, s_max: int, adaptive: bool) ->np.array:
    img_out = np.zeros(img_in.shape)
    if not adaptive:
        window = s_max
        for i in range(img_in.shape[0]):
            for j in range(img_in.shape[1]):
                matrix = make_mat(img_in, window, i, j)
                
                img_out[i][j] = find_median(matrix, window)
    else:
        for i in range(img_in.shape[0]):
            for j in range(img_in.shape[1]):
                window = 3
                while window<=s_max:
                    matrix = make_mat(img_in, window, i, j)
                    if not img_in[i][j] == find_max(matrix) and not img_in[i][j] == find_min(matrix):
                        img_out[i][j]=img_in[i][j]
                        break
                    pom = find_median(matrix, window)
                    if pom == find_max(matrix) or pom == find_min(matrix):
                        window=window+2
                        continue
                    img_out[i][j] = pom
                    break
    return img_out
        
def add_noise(img: np.array) -> np.array:
 
    # Getting the dimensions of the image
    row , col = img.shape
    max_value=255
    if type(img[0][0]) == np.float64:
        max_value=1
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = 50000
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to white
        img[y_coord][x_coord] = max_value
         
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = 50000
    for i in range(number_of_pixels):
       
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
         
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
         
        # Color that pixel to black
        img[y_coord][x_coord] = 0
         
    return img

if __name__ == "__main__":
    adaptive = False
    img_in = imread('zad4/gardos.jpg')
    img_in = color.rgb2gray(img_in)
    img_noise = add_noise(img_in)
    imsave("zad4/gardos_noise.jpg", img_noise, cmap = 'gray')
    img_out = dos_median(img_noise, 7, adaptive)
    if not adaptive:
        imsave("zad4/gardos_output_not_adaptive.jpg", img_out, cmap = 'gray')
    else:
        imsave("zad4/gardos_output_adaptive.jpg", img_out, cmap = 'gray')
        