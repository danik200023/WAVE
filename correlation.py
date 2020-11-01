import numpy as np
from math import sqrt, floor
import os
from os.path import join as pjoin
import subprocess
from scipy.io import wavfile

directories = os.listdir('C:/Users/Dmitry/Desktop/666/Wave_data/data/result_phonemes — копия')
for d in directories:
    files = os.listdir('C:/Users/Dmitry/Desktop/666/Wave_data/data/result_phonemes — копия/' + d)
    os.chdir('C:/Users/Dmitry/Desktop/666/result kartinOCHKA')
    os.mkdir('C:/Users/Dmitry/Desktop/666/result kartinOCHKA/' + d)
    os.chdir('C:/Users/Dmitry/Desktop/666/result kartinOCHKA/' + d)
    t = 0
    for f in files:
        wav_fname = pjoin('C:/Users/Dmitry/Desktop/666/Wave_data/data/result_phonemes — копия/' + d + '/' + f)
        np.set_printoptions(suppress=True, threshold=np.inf)  # threshold=np.inf
        wavrate, wavdata = wavfile.read(wav_fname)
        N = 25
        N1 = 30
        M = len(wavdata)
        if len(wavdata) > 1250:
            wavdata = np.delete(wavdata, wavdata[1250:len(wavdata) - 1])
        if len(wavdata) < 1250:
            while len(wavdata) != 1250:
                wavdata = np.append(wavdata, 0)
        M = len(wavdata)

        X = np.ndarray((N1 - 1, N))
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
        np.save(f, X)
        subprocess.call('/bin/cls')
        print('Current directory: ' + str(d))
        print(str(directories.index(d)) + '/' + str(len(directories)))
        print('Current file: ' + str(f))
        print(str(t) + '/' + str(len(files)))
        t += 1
