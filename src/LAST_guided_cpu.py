#!/usr/bin/env python3
# coding: utf-8

import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2.ximgproc as cvgd
import time
import timeit
import gc



def mean(x, r):
    return cv2.boxFilter(x, cv2.CV_64F, (r, r))


def Guidedfilter(im, p, r, eps):
    '''
    im: guide
    p: input
    r: size of kernel
    eps: regulation parameter
    '''
    mean_I = mean(im, r)
    mean_II = mean(im * im, r)
    mean_p = mean(p, r)
    mean_Ip = mean(im * p, r)

    cov_Ip = mean_Ip - mean_I * mean_p; # cov(x, y) = E(xy) - E(x)E(y)
    var_I = mean_II - mean_I * mean_I # var(x) = cov(x, x)

    # linear coeffs
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = mean(a, r)
    mean_b = mean(b, r)

    q = mean_a * im + mean_b
    return q


def compare(a, b):
    plt.figure(figsize = (15, 15))
    plt.subplot(1, 2, 1)
    plt.imshow(a)
    plt.subplot(1, 2, 2)
    plt.imshow(b)
    plt.show()


def fastGuidedfilter(im, p, r, eps, s): # s: ratio of resizing
    size = im.shape
    new_shape = (int(size[1] / s), int(size[0] / s))

    I_sub = cv2.resize(im, new_shape, interpolation = cv2.INTER_AREA)
    p_sub = cv2.resize(p, new_shape, interpolation = cv2.INTER_AREA)
    # r_sub = r / s

    mean_I = mean(I_sub, r)
    mean_p = mean(p_sub, r)
    mean_Ip = mean(I_sub * p_sub, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = mean(I_sub * I_sub, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = mean(a, r)
    mean_b = mean(b, r)
    mean_a = cv2.resize(mean_a, size[1::-1], interpolation = cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_b, size[1::-1], interpolation = cv2.INTER_LINEAR)

    q = mean_a * im + mean_b
    return q;


gc.collect()

I = cv2.cvtColor(cv2.imread("../data/cat.png"), cv2.COLOR_BGR2RGB)
#I = cv2.cvtColor(cv2.imread("../data/cat2.jpeg"), cv2.COLOR_BGR2RGB)
#I = cv2.cvtColor(cv2.imread("../data/bigcat.jpg"), cv2.COLOR_BGR2RGB)
#I = cv2.cvtColor(cv2.imread("../data/cave.bmp"), cv2.COLOR_BGR2RGB)
#I = cv2.cvtColor(cv2.imread("../data/lady.jpg"), cv2.COLOR_BGR2RGB)
#I = cv2.cvtColor(cv2.imread("../data/toy.bmp"), cv2.COLOR_BGR2RGB)
#I = cv2.cvtColor(cv2.imread("../data/tulips.bmp"), cv2.COLOR_BGR2RGB)
#I = cv2.cvtColor(cv2.imread("../data/ocean.jpg"), cv2.COLOR_BGR2RGB)
#I = cv2.cvtColor(cv2.imread("../data/cat3.jpg"), cv2.COLOR_BGR2RGB)
#I = cv2.cvtColor(cv2.imread("../data/cat4.jpg"), cv2.COLOR_BGR2RGB)
#I = cv2.cvtColor(cv2.imread("../data/cat5.jpg"), cv2.COLOR_BGR2RGB)

I_normed = np.float32(I / 255)

print(I.shape)
tot = 0
for i in range(100):
    #a = time.time()
    #a = time.process_time()
    a = time.perf_counter()
    #a = timeit.default_timer()
    #a = time.monotonic()
    # time.clock is deprecated

    #filtered = Guidedfilter(I_normed, I_normed, 8, 0.2 ** 2)
    filtered = fastGuidedfilter(I_normed, I_normed, 8, 0.2 ** 2, 4)
    #filtered = cvgd.guidedFilter(I_normed, I_normed, 8, 0.2 ** 2)

    #b = time.time()
    #b = time.process_time()
    b = time.perf_counter()
    #b = timeit.default_timer()
    #b = time.monotonic()
    # time.clock is deprecated

    tmp = ((b - a) * 1000.0)
    print(tmp)
    tot += tmp

print("mean: ", tot / 100.0)
print(I.shape)

compare(I, filtered)
