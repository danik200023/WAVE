import numpy as np
import os
from os.path import join as pjoin
from scipy.io import wavfile

directories = os.listdir('data/Dima/')
for d in directories:
    files = os.listdir('data/Dima/' + d)
    H = np.empty(len(files))
    t = 0

    for f in files:
        wav_fname = pjoin('data/Dima/' + d + '/' + f)
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
