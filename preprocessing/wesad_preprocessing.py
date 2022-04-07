import pickle
import numpy as np
import scipy.signal
#import tensorflow.keras as keras
from collections import Counter
import copy
from sklearn.model_selection import train_test_split
import torch


# from keras
def to_categorical(y, num_classes=None, dtype='float32'):
	y = np.array(y, dtype='int')
	input_shape = y.shape
	if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
		input_shape = tuple(input_shape[:-1])
	y = y.ravel()
	if not num_classes:
		num_classes = np.max(y) + 1
	n = y.shape[0]
	categorical = np.zeros((n, num_classes), dtype=dtype)
	categorical[np.arange(n), y] = 1
	output_shape = input_shape + (num_classes,)
	categorical = np.reshape(categorical, output_shape)
	return categorical


# Credits a Disarli
# Caricamento dei soggetti
ds = {
    "S2": pickle.load(open("datasets/WESAD/S2/S2.pkl", 'rb'), encoding='latin1'),
    "S3": pickle.load(open("datasets/WESAD/S3/S3.pkl", 'rb'), encoding='latin1'),
    "S4": pickle.load(open("datasets/WESAD/S4/S4.pkl", 'rb'), encoding='latin1'),
    "S5": pickle.load(open("datasets/WESAD/S5/S5.pkl", 'rb'), encoding='latin1'),
    "S6": pickle.load(open("datasets/WESAD/S6/S6.pkl", 'rb'), encoding='latin1'),
    "S7": pickle.load(open("datasets/WESAD/S7/S7.pkl", 'rb'), encoding='latin1'),
    "S8": pickle.load(open("datasets/WESAD/S8/S8.pkl", 'rb'), encoding='latin1'),
    "S9": pickle.load(open("datasets/WESAD/S9/S9.pkl", 'rb'), encoding='latin1'),
    "S10": pickle.load(open("datasets/WESAD/S10/S10.pkl", 'rb'), encoding='latin1'),
    "S11": pickle.load(open("datasets/WESAD/S11/S11.pkl", 'rb'), encoding='latin1'),
    "S13": pickle.load(open("datasets/WESAD/S13/S13.pkl", 'rb'), encoding='latin1'),
    "S14": pickle.load(open("datasets/WESAD/S14/S14.pkl", 'rb'), encoding='latin1'),
    "S15": pickle.load(open("datasets/WESAD/S15/S15.pkl", 'rb'), encoding='latin1'),
    "S16": pickle.load(open("datasets/WESAD/S16/S16.pkl", 'rb'), encoding='latin1'),
    "S17": pickle.load(open("datasets/WESAD/S17/S17.pkl", 'rb'), encoding='latin1')
}
print("Subjects loaded")

for s in ds.keys():
    # Concatenamento e ricampionamento a 32 Hz delle 17 features di WESAD
    X = np.concatenate([
        scipy.signal.resample(ds[s]['signal']['chest']['ACC'], len(ds[s]['signal']['wrist']['ACC'])),
        scipy.signal.resample(ds[s]['signal']['chest']['EDA'], len(ds[s]['signal']['wrist']['ACC'])),
        scipy.signal.resample(ds[s]['signal']['chest']['EMG'], len(ds[s]['signal']['wrist']['ACC'])),
        scipy.signal.resample(ds[s]['signal']['chest']['ECG'], len(ds[s]['signal']['wrist']['ACC'])),
        scipy.signal.resample(ds[s]['signal']['chest']['Resp'], len(ds[s]['signal']['wrist']['ACC'])),
        scipy.signal.resample(ds[s]['signal']['chest']['Temp'], len(ds[s]['signal']['wrist']['ACC'])),
        scipy.signal.resample(ds[s]['signal']['wrist']['ACC'], len(ds[s]['signal']['wrist']['ACC'])),
        scipy.signal.resample(ds[s]['signal']['wrist']['BVP'], len(ds[s]['signal']['wrist']['ACC'])),
        scipy.signal.resample(ds[s]['signal']['wrist']['EDA'], len(ds[s]['signal']['wrist']['ACC'])),
        scipy.signal.resample(ds[s]['signal']['wrist']['TEMP'], len(ds[s]['signal']['wrist']['ACC']))
        ], axis = 1)
    print(s, "Resampled")

    # Standardizzazione con media = 0 e deviazione standard = 1
    X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    print(s, "Standardized")

    # Ricampionamento delle etichette
    Y = scipy.signal.resample(ds[s]['label'], len(ds[s]['signal']['wrist']['ACC']))
    Y = np.around(Y)
    Y = abs(Y.astype(np.int_))
    print(s, "Labels")

    # Rimozione delle etichette inutilizzate
    X = X[(Y>0) & (Y<5)]
    Y = Y[(Y>0) & (Y<5)]
    print(s, "Cleaned")
    assert len(X) == len(Y)

    # Creazione delle sottosequenze da 100 elementi (3 secondi)
    count = 0
    prev = Y[0]
    LenSubsequences = []
    for elem in Y:
        if(elem != prev):
            LenSubsequences.append(count)
            count = 0
        count += 1
        prev = elem
    SubsequencesX = []
    SubsequencesY = []
    i = 0
    for elem in LenSubsequences:
        for j in range(0, elem, 100):
            if(j+100 <= elem):
                SubsequencesX.append(X[i+j:i+j+100])
                SubsequencesY.append(Y[i+j+50])
        i += elem

    assert len(SubsequencesX) == len(SubsequencesY)
    X_WES = (np.array(SubsequencesX, dtype = np.float32)).reshape(-1, 100, 14)

    # Le etichette 0 e 5, 6, 7 sono state tolte, quindi si spostano le rimanenti
    # da 1 - 4 a 0 - 3
    Y = np.array(SubsequencesY, dtype = np.int_) - 1
    #y_WES = to_categorical(Y, num_classes = 4)
    y_WES = Y

    # Selezione di 100 sottosequenze per ogni etichetta dal soggetto
    idx = np.argsort(Y)
    SubY, SubX = np.array(Y)[idx], np.array(X_WES)[idx]
    count = 0
    prev = SubY[0]
    SubsequencesX, X = [], []
    SubsequencesY, Y = [], []
    for i, elem in enumerate(SubY):
        if(elem != prev):
            count = 0
        count += 1
        if (count <= 100):
            SubsequencesX.append(SubX[i])
            SubsequencesY.append(elem)
        prev = elem

    X_WES = (np.array(SubsequencesX, dtype = np.float32))
    Y = np.array(SubsequencesY, dtype = np.int_)
    #y_WES = to_categorical(Y, num_classes = 4)
    y_WES = Y
    print(s, "Subsequences")

    # Salvataggio del soggetto
    with open("datasets/WESAD/splitted/X" + s + ".pkl", 'wb') as handle:
        pickle.dump(X_WES, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("datasets/WESAD/splitted/y" + s + ".pkl", 'wb') as handle:
        pickle.dump(y_WES, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Caricamento dei soggetti appena preprocessati
X, y = None, None
for S in ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]:
    Xs = pickle.load(open("datasets/WESAD/splitted/X" + S + ".pkl", 'rb'), encoding='latin1')
    ys = pickle.load(open("datasets/WESAD/splitted/y" + S + ".pkl", 'rb'), encoding='latin1')

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

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Preparazione del test set
_, Xts, _, yts = train_test_split(X, y, test_size = 0.25, train_size = 0.75, random_state=42)
print(str(Xts.shape) + " " + str(yts.shape))
train, targets = [], []
for i, elem in enumerate(Xts):
	train.append(elem)
	targets.append(yts[i])
with open("datasets/WESAD/splitted/Xts.pkl", 'wb') as handle:
    pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("datasets/WESAD/splitted/yts.pkl", 'wb') as handle:
    pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Rimozione dei dati usati nel test set dai vari soggetti
for S in ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]:
    Xs = pickle.load(open("datasets/WESAD/splitted/X" + S + ".pkl", 'rb'), encoding='latin1')
    ys = pickle.load(open("datasets/WESAD/splitted/y" + S + ".pkl", 'rb'), encoding='latin1')
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
        train.append(elem)
        targets.append(ys[i])
    with open("datasets/WESAD/splitted/X" + S + ".pkl", 'wb') as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("datasets/WESAD/splitted/y" + S + ".pkl", 'wb') as handle:
        pickle.dump(targets, handle, protocol=pickle.HIGHEST_PROTOCOL)
