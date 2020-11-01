import numpy as np
import os
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
