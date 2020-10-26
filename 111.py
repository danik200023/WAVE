import numpy as np
import tensorflow as tf
from keras.models import Sequential
import matplotlib.pyplot as plt
import PyQt5
from math import sqrt, floor
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import wavfile

wav_fname = pjoin('data/result_phonemes/A/00ca367f-979b-4ba7-a487-94344dda9a64.wav')
np.set_printoptions(suppress=True, threshold=np.inf) # threshold=np.inf
wavrate, wavdata = wavfile.read(wav_fname)
N = 16
N1 = 30
M = len(wavdata)
# print(M)
X = np.ndarray((N1 - 1, N))
# X = np.ndarray((floor(M / N) - 1, N))
for g in range(N1 - 1):
    for k in range(N):
        sum1 = np.float(0)
        sum2 = np.float(0)
        sum3 = np.float(0)
        sum4 = np.float(0)
        sum5 = np.float(0)
        for i in range(0 + g * floor(M / N1), N + g * floor(M / N1)):
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
plt.figure()
plt.imshow(X)
plt.colorbar()
plt.grid(False)
plt.show()
print(X)