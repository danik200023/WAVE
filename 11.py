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
R = np.empty(0)
for f in files:
    N = np.load('hueta/' + f)
    R = np.append(R, np.min(N))
print(np.mean(R))
print(np.median(R))
print(np.min(R))
print(np.max(R))