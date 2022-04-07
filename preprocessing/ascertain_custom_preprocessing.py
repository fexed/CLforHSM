
import numpy as np
#import tensorflow as tf
import scipy.io
import scipy.signal
import pickle
import copy
from sklearn.model_selection import train_test_split
import torch

# *** CUSTOM ASCERTAIN_custom ***

#Loading subjects
#Removing subjects with missing data
excluded_subjects = [44, 52]
#Excluding subjects data with poor accuracy & many nan values
n = 0
for i in range(37, 56):
    if(i not in excluded_subjects):
        ds_ECG, ds_EEG, ds_GSR = {}, {}, {}
        arousal, valence = [], []
        s = 'S' + str(i)
        clips_ECG = {}
        for j in range(1, 37):
          clips_ECG[('n_' + str(j))] = scipy.io.loadmat('datasets/ASCERTAIN/ECGData/Movie_P' + str(i) + '/ECG_Clip' + str(j))['Data_ECG'][:,1:]
        ds_ECG[s] = clips_ECG
        arousal.append(scipy.io.loadmat('datasets/ASCERTAIN/Dt_SelfReports.mat')['Ratings'][0][i])
        valence.append(scipy.io.loadmat('datasets/ASCERTAIN/Dt_SelfReports.mat')['Ratings'][1][i])
        clips_EEG = {}
        for j in range(1, 37):
          clips_EEG[('n_' + str(j))] = np.transpose(scipy.io.loadmat('datasets/ASCERTAIN/EEGData/Movie_P' + str(i) + '/EEG_Clip' + str(j))['ThisEEG'][:,1:])
        ds_EEG[s] = clips_EEG
        clips_GSR = {}
        for j in range(1, 37):
          clips_GSR[('n_' + str(j))] = scipy.io.loadmat('datasets/ASCERTAIN/GSRData/Movie_P' + str(i) + '/GSR_Clip' + str(j))['Data_GSR'][:,1:]
        ds_GSR[s] = clips_GSR
        Y = []
        for i in range(0, len(arousal)):
            for j in range(0, len(arousal[i])):
                if(arousal[i][j] > 3):
                    if(valence[i][j] > 0):
                        Y.append(0)
                    elif(valence[i][j] <= 0):
                        Y.append(1)
                elif(arousal[i][j] <= 3):
                    if(valence[i][j] > 0):
                        Y.append(2)
                    elif(valence[i][j] <= 0):
                        Y.append(3)
        for clip in ds_ECG[s].keys():
            ds_ECG[s][clip] = ds_ECG[s][clip][~np.isnan(ds_ECG[s][clip]).any(axis = 1)]
            ds_EEG[s][clip] = ds_EEG[s][clip][~np.isnan(ds_EEG[s][clip]).any(axis = 1)]
            ds_GSR[s][clip] = ds_GSR[s][clip][~np.isnan(ds_GSR[s][clip]).any(axis = 1)]
        for clip in ds_ECG[s].keys():
            ds_ECG[s][clip] = scipy.signal.resample(ds_ECG[s][clip], len(ds_EEG[s][clip]))
            ds_GSR[s][clip]= scipy.signal.resample(ds_GSR[s][clip], len(ds_EEG[s][clip]))
        ds = {}
        clips = {}
        j = 0
        for clip in ds_ECG[s].keys():
            clips['Clip' + str(j)] = np.concatenate([
            ds_ECG[s][clip],
            ds_EEG[s][clip],
            ds_GSR[s][clip],
            ], axis = 1)
            j += 1
        ds[s] = clips

        SubsequencesX, SubsequencesY = [], []
        count_0, count_1, count_2, count_3 = 0, 0, 0, 0
        j = 0
        for clip in ds[s].keys():
            if(((Y[j] == 0) & (count_0 < 84)) or ((Y[j] == 1) & (count_1 < 84)) or ((Y[j] == 2) & (count_2 < 84)) or ((Y[j] == 3) & (count_3 < 84))):
               length = len(ds[s][clip])
               for k in range(0, 10):
                   SubsequencesX.append(ds[s][clip][length-(160*(k+1)):length-(160*k),:])
                   if (Y[j] == 0):
                        SubsequencesY.append(0)
                   elif (Y[j] == 1):
                        count_1 += 1
                        SubsequencesY.append(1)
                   elif (Y[j] == 2):
                        count_2 += 1
                        SubsequencesY.append(2)
                   elif (Y[j] == 3):
                        count_3 += 1
                        SubsequencesY.append(3)
            j += 1


        X_ASC = (np.array(SubsequencesX, dtype = np.float32)).reshape(-1, 160, 17)
        y_ASC = np.array(SubsequencesY, dtype = np.int_)

        print("Shape of X:", X_ASC.shape)
        print("Shape of y:", y_ASC.shape)

        with open("datasets/ASCERTAIN_custom/splitted/XS" + str(n) + ".pkl", 'wb') as handle:
             pickle.dump(X_ASC, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("datasets/ASCERTAIN_custom/splitted/yS" + str(n) + ".pkl", 'wb') as handle:
             pickle.dump(y_ASC, handle, protocol=pickle.HIGHEST_PROTOCOL)
        n += 1

X, y = None, None
for S in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]:
    Xs = pickle.load(open("datasets/ASCERTAIN_custom/splitted/X" + S + ".pkl", 'rb'), encoding='latin1')
    ys = pickle.load(open("datasets/ASCERTAIN_custom/splitted/y" + S + ".pkl", 'rb'), encoding='latin1')

    if (X is None):
        X = copy.deepcopy(Xs)
        y = copy.deepcopy(ys)
    else:
        X = np.concatenate([X, Xs], axis = 0)
        y = np.concatenate([y, ys], axis = 0)
    del Xs
    del ys
    print("Loaded " + S)

print("Dataset loaded")

# Preparazione del test set
_, Xts, _, yts = train_test_split(X, y, test_size = 0.25, train_size = 0.75, random_state=42)
print(str(Xts.shape) + " " + str(yts.shape))
train, targets = [], []
for i, elem in enumerate(Xts):
	train.append(torch.from_numpy(elem))
	targets.append(yts[i])
with open("datasets/ASCERTAIN_custom/splitted/Xts.pkl", 'wb') as handle:
    pickle.dump(Xts, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("datasets/ASCERTAIN_custom/splitted/yts.pkl", 'wb') as handle:
    pickle.dump(yts, handle, protocol=pickle.HIGHEST_PROTOCOL)

for S in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]:
    Xs = pickle.load(open("datasets/ASCERTAIN_custom/splitted/X" + S + ".pkl", 'rb'), encoding='latin1')
    ys = pickle.load(open("datasets/ASCERTAIN_custom/splitted/y" + S + ".pkl", 'rb'), encoding='latin1')
    print(S + " " + str(Xs.shape) + " " + str(ys.shape), end = " -> ")
    j = []
    for xts in Xts:
        for i, xs in enumerate(Xs):
            if (xts == xs).all():
                j.append(i)

    Xs = np.delete(Xs, j, axis = 0)
    ys = np.delete(ys, j, axis = 0)

    print(str(Xs.shape) + " " + str(ys.shape))
    train, targets = [], []
    for i, elem in enumerate(Xs):
        train.append(torch.from_numpy(elem))
        targets.append(ys[i])
    with open("datasets/ASCERTAIN_custom/splitted/X" + S + ".pkl", 'wb') as handle:
        pickle.dump(Xs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("datasets/ASCERTAIN_custom/splitted/y" + S + ".pkl", 'wb') as handle:
        pickle.dump(ys, handle, protocol=pickle.HIGHEST_PROTOCOL)
