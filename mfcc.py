import os
import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc
import pickle

from scipy.io import wavfile
import torch
filepath = "/Users/tanikin/projct/TIMIT/train"
wavelist = []
spkr = []
dirnames = os.listdir(filepath)
i = 0

for dir in dirnames:
    if dir == '.DS_Store':
        continue
    for filename in os.listdir(os.path.join(filepath, dir)):
        if filename == '.DS_Store':
            continue
        i+=1
        spkr.append(i)
        for file in os.listdir(os.path.join(filepath, dir, filename)):
            if file == '.DS_Store':
                continue
            name, category = os.path.splitext(os.path.join(filepath, dir, filename, file))  # split the filename
            if category == '.wav':  # if wav file
                
                sample_rate, signal = wavfile.read(name+category)
                MFCC = mfcc(signal, samplerate=sample_rate, numcep=24
                            , nfilt=26, nfft=1024)

                mean_ = MFCC.mean(0)
                var_  = MFCC.var(0)
                MFCC_ = (MFCC- mean_)/var_ #normalise
                wavelist.append(MFCC_)#save MFCC 
                spkr.append(i)
label = []
for fi in spkr:
    label_ = np.zeros(max(spkr))
    label_[fi-1] = 1
    label.append(label_)# assign label to each voice

a = wavelist
wavlist = []
max_= len(max(wavelist,key = lambda x: len(x)))
for wav in a:
    b = np.zeros((max_,24))
    for i,j in enumerate(wav):
        b[i][0:len(j)] = j
    wavlist.append(b) # pad zeros to ensure each voice has the same frame number

    
with open('test1.pkl','wb') as f:
    pickle.dump([wavlist,label],f) # save the data and label into .pkl files
    
    
    