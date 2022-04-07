
import numpy as np
#import tensorflow as tf
import scipy.io
import scipy.signal
import pickle
import copy
from sklearn.model_selection import train_test_split
import torch


# Credits a Disarli
excluded_subjects = [44, 52]
# Si escludono soggetti i cui dati sono imprecisi o incompleti

# Caricamento soggetto per soggetto
n = 0
for i in range(37, 57):
    if(i not in excluded_subjects):
        print("S" + str(n), end = "")

        # Caricamento dati
        clips_ECG = {}
        for j in range(1, 37):
            clips_ECG[('n_' + str(j))] = scipy.io.loadmat('datasets/ASCERTAIN/ECGData/Movie_P' + str(i) + '/ECG_Clip' + str(j))['Data_ECG'][:,1:]
        clips_EEG = {}
        for j in range(1, 37):
            clips_EEG[('n_' + str(j))] = np.transpose(scipy.io.loadmat('datasets/ASCERTAIN/EEGData/Movie_P' + str(i) + '/EEG_Clip' + str(j))['ThisEEG'][:,1:])
        clips_GSR = {}
        for j in range(1, 37):
            clips_GSR[('n_' + str(j))] = scipy.io.loadmat('datasets/ASCERTAIN/GSRData/Movie_P' + str(i) + '/GSR_Clip' + str(j))['Data_GSR'][:,1:]

        arousal = scipy.io.loadmat('datasets/ASCERTAIN/Dt_SelfReports.mat')['Ratings'][0][i]
        valence = scipy.io.loadmat('datasets/ASCERTAIN/Dt_SelfReports.mat')['Ratings'][1][i]
        print(" loaded", end = "")

        # Creazione delle 4 classi:
        #   0 = arousal > 3 & valence > 0
        #   1 = arousal > 3 & valence <= 0
        #   2 = arousal <= 3 & valence > 0
        #   3 = arousal <= 3 & valence <= 0
        Y = []
        for j in range(len(arousal)):
            if(arousal[j] > 3):
                if(valence[j]> 0):
                    Y.append(0)
                elif(valence[j] <= 0):
                    Y.append(1)
            elif(arousal[j] <= 3):
                if(valence[j] > 0):
                    Y.append(2)
                elif(valence[j] <= 0):
                    Y.append(3)
        print(", targeted", end = "")

        # Rimozione dei valori incompleti
        for clip in clips_ECG.keys():
            clips_ECG[clip] = clips_ECG[clip][~np.isnan(clips_ECG[clip]).any(axis = 1)]
            clips_EEG[clip] = clips_EEG[clip][~np.isnan(clips_EEG[clip]).any(axis = 1)]
            clips_GSR[clip] = clips_GSR[clip][~np.isnan(clips_GSR[clip]).any(axis = 1)]
        print(", cleaned", end = "")

        # Ricampionamento a 32 Hz
        for clip in clips_ECG.keys():
            clips_ECG[clip] = scipy.signal.resample(clips_ECG[clip], len(clips_EEG[clip]))
            clips_GSR[clip]= scipy.signal.resample(clips_GSR[clip], len(clips_EEG[clip]))
        print(", resampled", end = "")

        # Concatenamento delle features di ASCERTAIN
        clips = {}
        j = 0
        for clip in clips_ECG.keys():
            clips['Clip' + str(j)] = np.concatenate([
            clips_ECG[clip],
            clips_EEG[clip],
            clips_GSR[clip],
            ], axis = 1)
            j += 1
        print(", merged", end = "")

        # Creazione delle sottosequenze lunghe 160 (5 secondi)
        SubsequencesX, SubsequencesY = [], []
        sbX_1, sbX_2, sbX_3, sbX_4 = [], [], [], []
        sby_1, sby_2, sby_3, sby_4 = [], [], [], []
        j = 0
        for clip in clips.keys():
           length = len(clips[clip])
           for k in range(0, 10):
               if (Y[j] == 0):
                    sbX_1.append(clips[clip][length-(160*(k+1)):length-(160*k),:])
                    sby_1.append(0)
               elif (Y[j] == 1):
                    sbX_2.append(clips[clip][length-(160*(k+1)):length-(160*k),:])
                    sby_2.append(1)
               elif (Y[j] == 2):
                    sbX_3.append(clips[clip][length-(160*(k+1)):length-(160*k),:])
                    sby_3.append(2)
               elif (Y[j] == 3):
                    sbX_4.append(clips[clip][length-(160*(k+1)):length-(160*k),:])
                    sby_4.append(3)
           j += 1

        print("", len(sby_1), len(sby_2), len(sby_3), len(sby_4), end = "")
        a = [len(sby_1), len(sby_2), len(sby_3), len(sby_4)]

        m = min(i for i in a if i > 0)

        sbX_1 = sbX_1[0:m]
        sby_1 = sby_1[0:m]
        sbX_2 = sbX_2[0:m]
        sby_2 = sby_2[0:m]
        sbX_3 = sbX_3[0:m]
        sby_3 = sby_3[0:m]
        sbX_4 = sbX_4[0:m]
        sby_4 = sby_4[0:m]

        SubsequencesX = sbX_1 + sbX_2 + sbX_3 + sbX_4
        SubsequencesY = sby_1 + sby_2 + sby_3 + sby_4
        print(" - ", m, len(SubsequencesX), len(SubsequencesY), end = "")
        Xs = (np.array(SubsequencesX, dtype = np.float32)).reshape(-1, 160, 17)
        #ys = to_categorical(np.array(SubsequencesY, dtype = np.float32), num_classes = 4)
        ys = np.array(SubsequencesY, dtype = np.int_)
        print(" and sampled (" + str(Xs.shape) + ")")

        # Salvataggio
        with open("datasets/ASCERTAIN/splitted/XS" + str(n) + ".pkl", 'wb') as handle:
            pickle.dump(Xs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("datasets/ASCERTAIN/splitted/yS" + str(n) + ".pkl", 'wb') as handle:
            pickle.dump(ys, handle, protocol=pickle.HIGHEST_PROTOCOL)
        n += 1

# Caricamento dei soggetti appena preprocessati
X, y = None, None
for S in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]:
    Xs = pickle.load(open("datasets/ASCERTAIN/splitted/X" + S + ".pkl", 'rb'), encoding='latin1')
    ys = pickle.load(open("datasets/ASCERTAIN/splitted/y" + S + ".pkl", 'rb'), encoding='latin1')

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
with open("datasets/ASCERTAIN/splitted/Xts.pkl", 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("datasets/ASCERTAIN/splitted/yts.pkl", 'wb') as handle:
    pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Rimozione da ogni soggetto dei dati che appaiono nel test set
for S in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16"]:
    Xs = pickle.load(open("datasets/ASCERTAIN/splitted/X" + S + ".pkl", 'rb'), encoding='latin1')
    ys = pickle.load(open("datasets/ASCERTAIN/splitted/y" + S + ".pkl", 'rb'), encoding='latin1')
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
    with open("datasets/ASCERTAIN/splitted/X" + S + ".pkl", 'wb') as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("datasets/ASCERTAIN/splitted/y" + S + ".pkl", 'wb') as handle:
        pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
