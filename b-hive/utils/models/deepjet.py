import numpy as np
import torch
import torch.nn as nn

from utils.models.abstract_base_models import Classifier
from utils.torch import DeepJetDataset
from utils.plotting.termplot import terminal_roc
from utils.models.helpers import DenseClassifier, InputProcess
from scipy.special import softmax

from rich.progress import (
	BarColumn,
	Progress,
	TaskProgressColumn,
	TextColumn,
	TimeElapsedColumn,
	TimeRemainingColumn,
)


class DeepJet(Classifier, nn.Module):
	n_jet = 4
	n_lepton = 2
	datasetClass = DeepJetDataset
	optimizerClass = torch.optim.Adam

	classes = {
		"ttbb": ["isttbb"],
		"ttcc": ["isttcc"],
		"ttother": ["isttbj", "isttcj", "isttother"],
	}

	global_features = [
		"nJet",
		"nbJet",
		"ncJet",
	]

	jet_features = [
		"Jet_pt",
		"Jet_eta",
		"Jet_phi",
		"Jet_mass",
		"Jet_drLep1",
		"Jet_drLep2",
		"Jet_btagDeepFlavB",
		"Jet_btagDeepFlavCvB",
		"Jet_btagDeepFlavCvL",
	]

	lepton_features = [
		"Lepton_pt",
		"Lepton_eta",
		"Lepton_phi",
		"Lepton_mass",
	]
	def __init__(self, feature_edges=[15, 415, 565, 613], **kwargs):
		super(DeepJet, self).__init__(**kwargs)

		self.feature_edges = np.array(feature_edges)
		self.InputProcess = InputProcess()
		self.DenseClassifier = DenseClassifier()

		self.global_bn = torch.nn.BatchNorm1d(3, eps=0.001, momentum=0.6)
		self.jet_lstm = torch.nn.LSTM(
			input_size=4, hidden_size=50, num_layers=1, batch_first=True
		) #input_size should be the same as final conv

		self.lepton_lstm = torch.nn.LSTM(
			input_size=4, hidden_size=50, num_layers=1, batch_first=True
		) #input_size should be the same as final conv

		self.jet_bn = torch.nn.BatchNorm1d(50, eps=0.001, momentum=0.6)
		self.lepton_bn = torch.nn.BatchNorm1d(50, eps=0.001, momentum=0.6)

		self.jet_dropout = nn.Dropout(0.1)
		self.lepton_dropout = nn.Dropout(0.1)

		#self.Linear = nn.Linear(100, len(self.classes))
		self.Linear = nn.Linear(25, len(self.classes))

	def forward(self, global_features, jet_features, lepton_features):
		global_features = self.global_bn(global_features)

		jet, lepton = self.InputProcess(jet_features, lepton_features)
		jet = self.jet_lstm(torch.flip(jet, dims=[1]))[0][:, -1]
		jet = self.jet_dropout(self.jet_bn(jet))

		lepton = self.lepton_lstm(torch.flip(lepton, dims=[1]))[0][:, -1]
		lepton = self.lepton_dropout(self.lepton_bn(lepton))

		fts = torch.cat((global_features, jet, lepton), dim=1)
		fts = self.DenseClassifier(fts)

		output = self.Linear(fts)

		return output

	def train_model(
		self,
		training_data,
		validation_data,
		directory,
		optimizer=None,
		device=None,
		nepochs=0,
		resume_epochs=0,
		**kwargs,
	):
		best_loss_val = np.inf
		loss_fn = nn.CrossEntropyLoss(reduction="none")
		loss_train = []
		acc_train = []
		loss_val = []
		acc_val = []

		scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
		# print("Initial ROC")

		# _, _ = self.validate_model(validation_data, loss_fn, device)
		for t in range(resume_epochs, nepochs):
			print("Epoch", t + 1, "of", nepochs)
			training_data.dataset.shuffleFileList()  # Shuffle the file list as mini-batch training requires it for regularisation of a non-convex problem
			loss_trainining, acc_training = self.update(
				training_data,
				loss_fn,
				optimizer=optimizer,
				scaler=scaler,
				device=device,
			)
			loss_train += loss_trainining 
			acc_train.append(acc_training)

			loss_validation, acc_validation = self.validate_model(validation_data, loss_fn, device)
			loss_val += loss_validation
			acc_val.append(acc_validation)


			torch.save(
				{
					"epoch": t,
					"model_state_dict": self.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"loss_train": loss_train,
					"acc_train": acc_train,
					"loss_val": loss_val,
					"acc_val": acc_val,
				},
				"{}/model_{}.pt".format(directory, t),
			)

			if np.mean(loss_validation) < best_loss_val:
				best_loss_val = np.mean(loss_validation)
				torch.save(
					{
						"epoch": t,
						"model_state_dict": self.state_dict(),
						"optimizer_state_dict": optimizer.state_dict(),
						"loss_train": loss_train,
						"acc_train": acc_train,
						"loss_val": loss_val,
						"acc_val": acc_val,
					},
					"{}/best_model.pt".format(directory),
				)

		return loss_train, loss_val, acc_train, acc_val,

	def predict_model(self, dataloader, device):
		self.eval()
		kinematics = []
		truths = []
		processes = []
		predictions = []
		for (
			global_features,
			jet_features,
			lepton_features,
			truth,
			weight,
			process,
		) in dataloader:
			# append jet_pt and jet_eta
			# this should be done differenlty in the future... avoid array slicing with magic numbers!
			with torch.no_grad():
				pred = self(
					*[
						feature.float().to(device)
						for feature in [
							global_features,
							jet_features,
							lepton_features,
						]
					]
				)
			kinematics.append(global_features[..., :2].cpu().numpy())
			truths.append(truth.cpu().numpy().astype(int))
			processes.append(process.cpu().numpy())
			predictions.append(pred.cpu().numpy().astype(int))

		predictions = np.concatenate(predictions)
		kinematics = np.concatenate(kinematics)
		truths = np.concatenate(truths)
		processes = np.concatenate(processes)
		return predictions, truths, kinematics, processes

	def update(
		self,
		dataloader,
		loss_fn,
		optimizer,
		scaler=None,
		device="cpu",
		verbose=True,
	):
		losses = []
		accuracy = 0.0
		self.train()

		with Progress(
			TextColumn("{task.description}"),
			TimeElapsedColumn(),
			BarColumn(bar_width=None),
			TaskProgressColumn(),
			TimeRemainingColumn(),
			TextColumn("0/? its"),
			expand=True,
		) as progress:
			N = 0
			task = progress.add_task("Training...", total=dataloader.nits_expected)
			print("entering traing loop")
			for (
				global_features,
				jet_features,
				lepton_features,
				truth,
				weight,
				process,
			) in dataloader:
				with torch.autocast(
					device_type=device, enabled=True if device == "cuda" else False
				):  # We select either cuda float16 mixed precision or cpu float32 as LSTMs does not accept bfloat16
					pred = self.forward(
						*[
							feature.float().to(device)
							for feature in [
								global_features,
								jet_features,
								lepton_features,
							]
						]
					)
					loss = loss_fn(pred, truth.type(torch.LongTensor).to(device)).mean()

				if scaler != None:
					optimizer.zero_grad(set_to_none=True)
					scaler.scale(loss).backward()
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
					scaler.step(optimizer)
					scaler.update()
				else:
					optimizer.zero_grad(set_to_none=True)
					loss.backward()
					optimizer.step()

				losses.append(loss.item())
				accuracy += (
					(pred.argmax(1) == truth.to(device)).type(torch.float).sum().item()
				)
				
				N += len(pred)
			
				progress.update(
					task, advance=1, description=f"Training...   | Loss: {loss:.2f}"
				)
				progress.columns[-1].text_format = "{}/{} its".format(
					N // dataloader.batch_size,
					(
						"?"
						if dataloader.nits_expected == len(dataloader)
						else f"~{dataloader.nits_expected}"
					),
				)
			progress.update(task, completed=dataloader.nits_expected)
		dataloader.nits_expected = N // dataloader.batch_size
		accuracy /= N
		print("  ", f"Average loss: {np.array(losses).mean():.4f}")
		print("  ", f"Average accuracy: {float(100*accuracy):.4f}")
		return losses, accuracy

	def validate_model(self, dataloader, loss_fn, device="cpu", verbose=True):
		losses = []
		accuracy = 0.0
		self.eval()

		#predictions = np.empty((0, 6))
		predictions = np.empty((0, 3))
		truths = np.empty((0))
		processes = np.empty((0))

		with Progress(
			TextColumn("{task.description}"),
			TimeElapsedColumn(),
			BarColumn(bar_width=None),
			TaskProgressColumn(),
			TimeRemainingColumn(),
			TextColumn("0/? its"),
			expand=True,
		) as progress:
			N = 0
			task = progress.add_task("Validation...", total=dataloader.nits_expected)
			for (
				global_features,
				jet_features,
				lepton_features,
				truth,
				weight,
				process,
			) in dataloader:
				with torch.no_grad():
					pred = self.forward(
						*[
							feature.float().to(device)
							for feature in [
								global_features,
								jet_features,
								lepton_features,
							]
						]
					)
					loss = loss_fn(pred, truth.type(torch.LongTensor).to(device)).mean()
					losses.append(loss.item())

					accuracy += (
						(pred.argmax(1) == truth.to(device))
						.type(torch.float)
						.sum()
						.item()
					)
					predictions = np.append(predictions, pred.to("cpu").numpy(), axis=0)
					truths = np.append(truths, truth.to("cpu").numpy(), axis=0)
					processes = np.append(processes, process.to("cpu").numpy(), axis=0)
				N += global_features.size(dim=0)
				progress.update(
					task, advance=1, description=f"Validation... | Loss: {loss:.2f}"
				)
				progress.columns[-1].text_format = "{}/{} its".format(
					N // dataloader.batch_size,
					(
						"?"
						if dataloader.nits_expected == len(dataloader)
						else f"~{dataloader.nits_expected}"
					),
				)
			progress.update(task, completed=dataloader.nits_expected)
		dataloader.nits_expected = N // dataloader.batch_size
		accuracy /= N
		print("  ", f"Validation loss: {np.array(losses).mean():.4f}")
		print("  ", f"Validation accuracy: {float(accuracy):.4f}")

		if verbose:
			print("Printing terminal ROC")
			terminal_roc(predictions, truths, title="Validation ROC")

		return losses, float(accuracy)

	def calculate_roc_list(
		self,
		predictions,
		truth,
	):
		if np.abs(np.mean(np.sum(predictions, axis=-1)) - 1) > 1e-3:
			predictions = softmax(predictions, axis=-1)

		ttbb = (truth == 0)
		ttcc = (truth == 2)
		ttother = (truth == 1) | (truth == 3) | (truth == 4)
		summed_proc = ttbb + ttbb + ttother

		ttbb_pred = predictions[:, 0].sum(axis=1)
		ttcc_pred = predictions[:, 2]
		ttother_pred = predictions[:, [1, 3, 4]].sum(axis=1)

		bvsall = np.where(
			(ttbb_pred + ttother_pred + ttcc_pred) > 0, (ttbb_pred) / (ttbb_pred + ttother_pred + ttcc_pred), -1
		)

		ttbb_veto = (truth != 0) & (summed_proc != 0)
		ttcc_veto = (truth != 2) & (summed_proc != 0)
		ttother_veto = (truth != 1) & (truth != 3) (truth != 4) & (summed_proc != 0)
		no_veto = np.ones(ttbb_veto.shape, dtype=np.bool)

		labels = ["ttbbvsttother", "ttbbvsttcc", "ttccvsttbb", "ttccvsttother", "ttbbvsall"]
		discs = [ttbbvsttother, ttbbvsttcc, ttccvsttbb, ttccvsttother, ttbbvsall]
		vetos = [ttcc_veto, ttother_veto, ttother_veto, ttbb_veto, no_veto]
		truths = [ttbb_jets, ttbb_jets, ttcc_jets, ttcc_jets, ttbb_jets]
		xlattbbels = [
			"ttbb-identifittccation",
			"ttbb-identifittccation",
			"ttcc-identifittccation",
			"ttcc-identifittccation",
			"ttbb-identifittccation",
		]
		ylabels = ["ttother mis-id.", "ttcc mis-id", "ttbb mis-id.", "ttother mis-id.", "mis-id."]

		return discs, truths, vetos, labels, xlabels, ylabels


class DeepJetHLT(DeepJet):
	n_jet = 4
	n_lepton = 2
	global_features = [
		"nJet",
		"nbJet",
		"ncJet",
	]
	jet_features = [
		"Jet_pt",
		"Jet_eta",
		"Jet_phi",
		"Jet_mass",
		"Jet_drLep1",
		"Jet_drLep2",
	]
	lepton_features = [
		"Lepton_pt",
		"Lepton_eta",
		"Lepton_phi",
		"Lepton_mass",
	]
