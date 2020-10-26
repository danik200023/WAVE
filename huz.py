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
files = os.listdir('hueta')
dir = os.listdir('data/result_phonemes — копия')
for g in range(len(files)):
    filess = os.listdir('data/result_phonemes — копия/' + dir[g])
    pis = np.load('hueta/' + files[g])
    if len(pis) > 10:
        percent = len(pis)//10
        sort = np.array([])
        sort = np.append(sort, pis)
        sort.sort()
        sort = np.append(sort[0:percent], sort[len(sort) - percent:len(sort)])
        for s in sort:
            index = np.where(pis == s)
            os.remove('data/result_phonemes — копия/' + dir[g] + '/' + filess[index[0][0]])
            filess = np.delete(filess, index[0][0])
            pis = np.delete(pis, index[0][0])
    print(dir[g])
