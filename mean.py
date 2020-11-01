import numpy as np
import os

files = os.listdir('hueta')
R = np.empty(0)
for f in files:
    N = np.load('hueta/' + f)
    R = np.append(R, np.min(N))
print(np.mean(R))
print(np.median(R))
print(np.min(R))
print(np.max(R))
