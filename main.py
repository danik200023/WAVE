import numpy as np
import tensorflow as tf
from keras.models import Sequential
#import torch
import matplotlib.pyplot as plt
import PyQt5
from math import sqrt, floor
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import wavfile

directories = os.listdir('data/result_phonemes/')
for d in directories:
    files = os.listdir('data/result_phonemes/' + d)
    H = np.empty(len(files), dtype=np.ndarray)
    t = 0

    for f in files:
        wav_fname = pjoin('data/result_phonemes/' + d + '/' + f)
        # np.set_printoptions(suppress=True)# threshold=np.inf
        wavrate, wavdata = wavfile.read(wav_fname)
        N = 50
        M = len(wavdata)
        X = np.ndarray((floor(M / N) - 1, N))
        for g in range(floor(M / N) - 1):
            for k in range(N):
                sum1 = np.int64(0)
                sum2 = np.int64(0)
                sum3 = np.int64(0)
                sum4 = np.int64(0)
                sum5 = np.int64(0)
                for i in range(0 + g * N, N + g * N):
                    sum1 += np.float64(wavdata[i]) * np.float64(wavdata[i + k])
                    sum2 += wavdata[i]
                    sum3 += wavdata[i + k]
                    sum4 += wavdata[i] ** 2
                    sum5 += wavdata[i + k] ** 2
                cov = sum1 / N - ((sum2 / N) * (sum3 / N))
                sum4 /= N
                sum5 /= N
                sum2 /= N
                sum3 /= N
                R = cov / (sqrt(sum4 - sum2 ** 2) * sqrt(sum5 - sum3 ** 2))
                X[g][k] = R
        os.system('cls')
        print('Current directory:' + str(d))
        print(str(directories.index(d)) + '/' + str(len(directories)))
        print('Current file: ' + str(f))
        print(str(t) + '/' + str(len(files)))
        H[t] = X #ПОСОСИ
        t += 1
        
    np.save(d, H)
