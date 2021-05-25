import numpy as np
import matplotlib.pyplot as plt
import math
import os
from os.path import join as pjoin
from scipy.io import wavfile

pycharm = True
if pycharm:
    import pyautogui
input_directory = 'C:/Users/Dmitry/Desktop/Python/WAVE/data/speech_units/'
output_directory = 'C:/Users/Dmitry/Desktop/Python/WAVE/data/speech_units_result/'
directories = os.listdir(input_directory)
local = 40
wstrings_in_acp = 96
lacpr = 16
step = 1
otstup = 40
mnogitel_proreditel = 1
width_ = 96
c = 0
for d in directories:
    files = os.listdir(input_directory + d)
    H = np.empty(len(files), dtype=np.ndarray)
    os.chdir(output_directory)
    os.mkdir(d)
    os.chdir(output_directory + d)
    t = 0
    for f in files:
        if f.find(".png") == -1:
            wav_fname = pjoin(input_directory + d + '/' + f)
            wavrate, wavdata = wavfile.read(wav_fname)
            acp = np.empty(0)
            N = len(wavdata)
            p = local
            ds, dd, ms, md = 0, 0, 0, 0
            while p + wstrings_in_acp < N - lacpr * step - local - (width_ + otstup + 1) * mnogitel_proreditel:

                plocal = p - local
                maxlocalpoint = wavdata[plocal]
                for i in range(p - local, p + local - 1):
                    if wavdata[i] > maxlocalpoint:
                        maxlocalpoint = wavdata[i]
                        plocal = i
                        p = plocal

                for i in range(1 + otstup, width_ + otstup + 1):
                    for j in range(p, p + lacpr * step, step):
                        ms = ms + wavdata[j]
                        md = md + wavdata[j + i * mnogitel_proreditel]
                    md = md / lacpr
                    ms = ms / lacpr
                    for j in range(p, p + lacpr * step, step):
                        ds = ds + (wavdata[j] - ms) * (wavdata[j] - ms)
                        dd = dd + (wavdata[j + i * mnogitel_proreditel] - md) * (wavdata[j + i * mnogitel_proreditel] - md)
                    dd = dd / lacpr
                    ds = ds / lacpr
                    for j in range(p, p + lacpr * step, step):
                        c = c + (wavdata[j] - ms) * (wavdata[j + i * mnogitel_proreditel] - md)
                    c = c / lacpr
                    if math.sqrt(ds * dd) == 0:
                        acp = np.append(acp, 0)
                    else:
                        acp = np.append(acp, c / math.sqrt(ds * dd))
                    c = 0
                    md, dd, ms = 0, 0, 0
                p += wstrings_in_acp
            acp = acp.reshape(int(len(acp)/width_), width_)
            plt.imsave(f[:-4] + ".png", acp)
            os.system('cls' if os.name == 'nt' else 'clear')
            if pycharm:
                pyautogui.hotkey('ctrl', 'l')
            print('Current directory: ' + str(d))
            print(str(directories.index(d)) + '/' + str(len(directories)))
            print('Current file: ' + str(f))
            print(str(t) + '/' + str(len(files)))
            t += 1
        else:
            pass

