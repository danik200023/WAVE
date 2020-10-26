import numpy as np
import tensorflow as tf
from keras.models import Sequential
import torch
import matplotlib.pyplot as plt
import PyQt5
from math import sqrt, floor
import os
from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import wavfile

directories = os.listdir('data/Dima/')
for d in directories:
    files = os.listdir('data/Dima/' + d)
    H = np.empty(len(files))
    t = 0

    for f in files:
        wav_fname = pjoin('data/Dima/' + d + '/' + f)
        # np.set_printoptions(suppress=True)# threshold=np.inf
        wavrate, wavdata = wavfile.read(wav_fname)
        M = len(wavdata)
        H[t] = M
        os.system('cls')
        print('Current directory: ' + str(d))
        print(str(directories.index(d)) + '/' + str(len(directories)))
        print('Current file: ' + str(f))
        print(str(t) + '/' + str(len(files)))
        t += 1
    np.save(d, H)
