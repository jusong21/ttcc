import os
from typing import List

import awkward as ak
import hist
import numpy as np
from coffea import processor


class DataPreprocessing_BaseClass(processor.ProcessorABC):
	n_jet = 4
<<<<<<< HEAD
=======
	n_lepton = 2
>>>>>>> develop
#	n_cpf = 50
#	n_npf = 50
#	n_vtx = 5

	def __init__(
		self,
		output_directory=None,
		bins_pt: List = None,
		bins_eta: List = None,
		prefix="",
		precision=np.float32,
		global_features: List[str] = None,
		jet_features: List[str] = None,
<<<<<<< HEAD
#		cpf_candidates: List[str] = None,
#		npf_candidates: List[str] = None,
#		vtx_features: List[str] = None,
		truths: List[str] = None,
		processes: List[str] = None,
		n_jet_candidates=4,
#		n_cpf_candidates=50,
#		n_npf_candidates=50,
#		n_vtx_features=5,
=======
		lepton_features: List[str] = None,
		truths: List[str] = None,
		processes: List[str] = None,
		n_jet_candidates=4,
		n_lepton_candidates=2,
>>>>>>> develop
	):
		self._accumulator = processor.dict_accumulator({})
		self.bins_eta = bins_eta
		self.bins_pt = bins_pt
		self.output_dir = output_directory
		self.precision = precision
		self.prefix = prefix
		self.processes = processes
<<<<<<< HEAD
#		self.cpf = cpf_candidates
#		self.npf = npf_candidates
#		self.vtx = vtx_features
		self.global_features = global_features
		self.jet_features = jet_features
		self.truths = truths
		self.n_jet = n_jet_candidates
#		self.n_cpf = n_cpf_candidates
#		self.n_npf = n_npf_candidates
#		self.n_vtx = n_vtx_features
=======
		self.global_features = global_features
		self.jet_features = jet_features
		self.lepton_features = lepton_features
		self.truths = truths
		self.n_jet = n_jet_candidates
		self.n_lepton = n_lepton_candidates
>>>>>>> develop
		if self.processes is None:
			self.processes = []

		self.truth_hists = {}

		for truth in self.truths:
			self.truth_hists[truth] = (
				hist.Hist.new.Variable(self.bins_pt, name="pt")
				.Variable(self.bins_eta, name="eta")
				.Int64()
			)
		self.setFeatureNamesAndEdges()

	def setFeatureNamesAndEdges(self):
		feature_edges = []
		feature_names = []

		feature_names.append(self.global_features)
		feature_edges.append(len(feature_names))
<<<<<<< HEAD
#		feature_edges.append(feature_edges[-1] + len(self.cpf) * self.n_cpf)
#		feature_names.extend(self.cpf)
#		feature_edges.append(feature_edges[-1] + len(self.npf) * self.n_npf)
#		feature_names.extend(self.npf)
#		feature_edges.append(feature_edges[-1] + len(self.vtx) * self.n_vtx)
#		feature_names.extend(self.vtx)
		feature_edges.append(feature_edges[-1] + len(self.jet_features) * self.n_jet)
		feature_names.extend(self.jet_features)
=======
		feature_edges.append(feature_edges[-1] + len(self.jet_features) * self.n_jet)
		feature_names.extend(self.jet_features)
		feature_edges.append(feature_edges[-1] + len(self.lepton_features) * self.n_lepton)
		feature_names.extend(self.lepton_features)
>>>>>>> develop
		feature_names.append("truths")
		feature_names.extend(self.truths)
		feature_names.append("process")

		self.feature_edges = feature_edges
		self.features = feature_names

	def saveOutput(self, output_location, output):
		pass

	@property
	def accumulator(self):
		return self._accumulator

	def callColumnAccumulator(self, output, events, **kwargs):
		pass

	def process(self, events):
		dataset = events.metadata["dataset"]

		start = events.metadata["entrystart"]
		stop = events.metadata["entrystop"]
		filename = "_".join(events.metadata["filename"].split("/")[1:]).split(".")[0]
<<<<<<< HEAD
		print('filename: ', filename)
=======
		#print('filename: ', filename)
>>>>>>> develop

		# assign process number
		if self.processes == ["default"]:
			proc_flag = 0
		else:
			proc_flag = -1
			for i, proc in enumerate(self.processes):
				if proc in dataset:
					proc_flag = i

		output = self.accumulator
		output_location_list = []

		(
			global_arr,
			jet_arr,
<<<<<<< HEAD
#			cpf_arr,
#			npf_arr,
#			vtx_arr,
=======
			lepton_arr,
>>>>>>> develop
			truth,
			process,
		) = self.callColumnAccumulator(
			output, events, proc_flag, filename=events.metadata["filename"]
		)

		for truth_label in self.truths:
			self.truth_hists[truth_label].fill(
				global_arr["nJet"][truth[truth_label]],
				global_arr["nbJet"][truth[truth_label]],
##				global_arr["jet_pt"][truth[truth_label]],
##				global_arr["jet_eta"][truth[truth_label]],
			)

		output_location = os.path.join(
			self.output_dir, f"{self.prefix}{dataset}_{filename}_{start}_{stop}.npz"
		)

		output_location_list.append(output_location)

		self.saveOutput(
<<<<<<< HEAD
			#output_location, global_arr, cpf_arr, npf_arr, vtx_arr, truth, process
			output_location, global_arr, jet_arr, truth, process
			#output_location, global_arr, jet_arr, truth, 
=======
			output_location, global_arr, jet_arr, lepton_arr, truth, process
>>>>>>> develop
		)
		ret = {}
		ret["output_location"] = output_location_list
		for label, hist in self.truth_hists.items():
			ret[label] = hist
		return ret

	def postprocess(self, accumulator):
		pass
