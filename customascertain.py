import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import copy


# For each subject
#X, y = None, None
#for S in ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S13", "S14", "S15", "S16", "S17"]:
	#Xs = pickle.load(open("datasets/WESAD/splitted/X" + S + ".pkl", 'rb'), encoding='latin1')
	#ys = pickle.load(open("datasets/WESAD/splitted/y" + S + ".pkl", 'rb'), encoding='latin1')


class ASCERTAINTrSet(Dataset):
	def __init__(self, pair, transform=None):
		self.root_dir = "datasets/ASCERTAIN_custom/splitted/"
		self.transform = transform
		self.pair = pair
		subjs = [("S0", "S1"), ("S2", "S3"), ("S4", "S5"), ("S6", "S7"), ("S8", "S9"), ("S10", "S11"), ("S12", "S13"), ("S14", "S15"), ("S16", "")]
		couple = subjs[self.pair]

		X, y = None, None
		for S in couple:
			if S != "":
				Xs = pickle.load(open(self.root_dir + "X" + S + ".pkl", 'rb'), encoding='latin1')
				ys = pickle.load(open(self.root_dir + "y" + S + ".pkl", 'rb'), encoding='latin1')
				if (X is None):
					X = copy.deepcopy(Xs)
					y = copy.deepcopy(ys)
				else:
					X = np.concatenate([X, Xs], axis = 0)
					y = np.concatenate([y, ys], axis = 0)
				del Xs
				del ys
		if self.transform:
			X = self.transform(X)

		self.data = X
		self.targets = y

	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx):
		data, target = self.data[idx], self.targets[idx]
		return data, target


class ASCERTAINTsSet(Dataset):
	def __init__(self, transform=None):
		self.root_dir = "datasets/ASCERTAIN_custom/splitted/"
		self.transform = transform
		Xts = pickle.load(open(self.root_dir + "Xts.pkl", 'rb'), encoding='latin1')
		yts = pickle.load(open(self.root_dir + "yts.pkl", 'rb'), encoding='latin1')
		if self.transform:
			Xts = self.transform(Xts)
		self.data = Xts
		self.targets = yts

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		data, target = self.data[idx], self.targets[idx]
		return data, target
