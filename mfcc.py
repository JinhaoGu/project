import os
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc, logfbank
import pickle
import dill

from scipy.io import wavfile
import torch
filepath = "/Users/tanikin/projct/TIMIT/test"
wavelist = []
dirnames = os.listdir(filepath)
i = 1
for dir in dirnames:
    if dir == '.DS_Store':
        continue
    for filename in os.listdir(os.path.join(filepath, dir)):
        if filename == '.DS_Store':
            continue
        i+=1
        for file in os.listdir(os.path.join(filepath, dir, filename)):
            if file == '.DS_Store':
                continue
            name, category = os.path.splitext(os.path.join(filepath, dir, filename, file))  # 分解文件扩展名
            if category == '.wav':  # 若文件为wav音频文件
                # wavelist.append(name+category)
                sample_rate, sig = wavfile.read(name+category)
                MFCC = mfcc(sig, samplerate=sample_rate, numcep=24
                            , nfilt=26, nfft=1024)

#print(wavelist)

print(MFCC[0])
'''''
for wav in wavelist:

    sample_rate, sig = wavfile.read(wavelist[0])
    i+=1


with open('test.pkl','wb') as f:
    pickle.dump([sample_rate,sig],f)


with open('test.pkl','rb') as f:
    o,j = pickle.load(f)


'''