from wesad import WESADTrSet, WESADTsSet
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models import SimpleMLP
from avalanche.training.strategies import Naive, Cumulative, LwF, EWC, JointTraining, GEM, Replay
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics, gpu_usage_metrics
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
import pickle
import torch.nn as nn
import torch
import numpy as np
import sys
import time


class Classifier(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim, layers=1):
		super().__init__()
		self.rnn = nn.GRU(in_dim, hidden_dim, layers, batch_first=True)
		self.clf = nn.Linear(hidden_dim, out_dim)

	def forward(self, x):
		# batch_size first
		x, _ = self.rnn(x)  # _ to ignore state
		x = x[:, -1]  # last timestep for classfication
		return self.clf(x)


import warnings
warnings.filterwarnings("ignore")


def train_wesad(strat, i=""):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	scenario = dataset_benchmark(
		[AvalancheDataset(WESADTrSet(pair=0), task_labels=1),
		AvalancheDataset(WESADTrSet(pair=1), task_labels=2),
		AvalancheDataset(WESADTrSet(pair=2), task_labels=3),
		AvalancheDataset(WESADTrSet(pair=3), task_labels=4),
		AvalancheDataset(WESADTrSet(pair=4), task_labels=5),
		AvalancheDataset(WESADTrSet(pair=5), task_labels=6),
		AvalancheDataset(WESADTrSet(pair=6), task_labels=7),
		AvalancheDataset(WESADTrSet(pair=7), task_labels=8)],
		[AvalancheDataset(WESADTsSet(), task_labels=0)]
	)

	tb_logger = TensorboardLogger()
	text_logger = TextLogger(open('wesadlog.txt', 'a'))
	int_logger = InteractiveLogger()

	eval_plugin = EvaluationPlugin(
	    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
	    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
	    timing_metrics(epoch=True, epoch_running=True),
	    forgetting_metrics(experience=True, stream=True),
	    cpu_usage_metrics(experience=True),
	    gpu_usage_metrics(0, experience=True),
	    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
	    loggers=[text_logger]
	)

	es = EarlyStoppingPlugin(patience=25, val_stream_name="train_stream")

	results = []
	model = Classifier(in_dim=14, hidden_dim=18, out_dim=4, layers=2)

	if (strat == "naive"):
		print("Naive continual learning")
		strategy = Naive(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
	elif (strat == "offline"):
		print("Offline learning")
		strategy = JointTraining(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
	elif (strat == "replay"):
		print("Replay training")
		strategy = Replay(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, mem_size=70, train_mb_size=70)  #25% of WESAD
	elif (strat == "cumulative"):
		print("Cumulative continual learning")
		strategy = Cumulative(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
	elif (strat == "lwf"):
		print("LwF continual learning")
		strategy = LwF(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, alpha=0.5, temperature=1)
	elif (strat == "ewc"):
		print("EWC continual learning")
		torch.backends.cudnn.enabled = False
		strategy = EWC(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, ewc_lambda=0.99)
	elif (strat == "episodic"):
		print("Episodic continual learning")
		strategy = GEM(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, patterns_per_exp=70)

	thisresults = []

	print(i + ".")
	start = time.time()
	if strat == "offline":
		res = strategy.train(scenario.train_stream)
		r = strategy.eval(scenario.test_stream)
		thisresults.append({"loss":r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
						"acc":(float(r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"])*100),
						"forg":r["StreamForgetting/eval_phase/test_stream"],
						"all":r})
		results.append({"strategy":strat,
						"finalloss":r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
						"finalacc":r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"],
						"results":thisresults})
	else:
		for experience in scenario.train_stream:
			res = strategy.train(experience)
			r = strategy.eval(scenario.test_stream)
			thisresults.append({"loss":r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
							"acc":(float(r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"])*100),
							"forg":r["StreamForgetting/eval_phase/test_stream"],
							"all":r})
		results.append({"strategy":strat,
						"finalloss":r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
						"finalacc":r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"],
						"results":thisresults})
	elapsed = time.time() - start
	results.append({"time":elapsed})
	with open("results/wesad_" + strat + "_results" + i + ".pkl", "wb") as outfile:
		pickle.dump(results, outfile)
	print("\t" + str(elapsed) + " seconds")


for i in range(5):
	train_wesad(sys.argv[1].strip(), str(i))
