import collections, awkward as ak, numpy as np
import os
import uproot
from coffea import processor
from coffea.analysis_tools import Weights

# functions to load SFs, corrections
from BTVNanoCommissioning.utils.correction import (
	load_lumi,
	load_SF,
	met_filters,
	HLTSFs,
	eleSFs,
	muSFs,
	puwei,
	btagSFs,
	JME_shifts,
	Roccor_shifts,
)

# user helper function
from BTVNanoCommissioning.helpers.func import (
	flatten,
	update,
	uproot_writeable,
	dump_lumi,
)
from BTVNanoCommissioning.helpers.update_branch import missing_branch

## load histograms & selctions for this workflow
from BTVNanoCommissioning.utils.histogrammer import histogrammer
from BTVNanoCommissioning.utils.selection import (
	jet_id, mu_idiso, 
	ele_cuttightid,
	btag_wp_dict,
	btag_wp,
)
from BTVNanoCommissioning.helpers.ttcc2L2Nu_helper import (
	sel_HLT,
	to_bitwise_trigger,
)

class NanoProcessor(processor.ProcessorABC):
	def __init__(
		self,
		year="2022",
		campaign="Summer22Run3",
		name="",
		isSyst=False,
		isArray=True,
		noHist=False,
		chunksize=75000,
		isTTbar=False
	):
		self._year = year
		self._campaign = campaign
		self.name = name
		self.isSyst = isSyst
		self.isArray = isArray
		self.noHist = noHist
		self.lumiMask = load_lumi(self._campaign)
		self.chunksize = chunksize
		## Load corrections
		self.SF_map = load_SF(self._campaign)
		self.isTTbar = isTTbar
		

	@property
	def accumulator(self):
		return self._accumulator

	## Apply corrections on momentum/mass on MET, Jet, Muon
	def process(self, events):
		print("**************************************")
		print("This is ", self._campaign, "ttcc dilepton channel producer")
		print("**************************************")
		#isRealData = not hasattr(events, "genWeight")
		dataset = events.metadata["dataset"]
#		events = missing_branch(events)
#		shifts = []
#		if "JME" in self.SF_map.keys():
#			syst_JERC = True if self.isSyst != None else False
#			if self.isSyst == "JERC_split":
#				syst_JERC = "split"
#			shifts = JME_shifts(
#				shifts, self.SF_map, events, self._campaign, isRealData, syst_JERC
#			)
#		else:
#			if "Run3" not in self._campaign:
#				shifts = [
#					({"Jet": events.Jet, "MET": events.MET, "Muon": events.Muon}, None)
#				]
#			else:
#				shifts = [
#					({"Jet": events.Jet, "MET": events.PuppiMET, "Muon": events.Muon,}, None,)
#				]
#		if "roccor" in self.SF_map.keys():
#			shifts = Roccor_shifts(shifts, self.SF_map, events, isRealData, False)
#		else:
#			shifts[0][0]["Muon"] = events.Muon
#
#		for collections, name in shifts:
#			print('n ', name)
#			print('n ', name)
#			print('n ', name)
#			print('col', collections)
#			print('col', collections)
#
		return processor.accumulate(
			#self.process_shift(update(events, collections), name)
			self.process_shift(events)
			#for collections, name in shifts
		)

	#def process_shift(self, events, shift_name):
	def process_shift(self, events):
		dataset = events.metadata["dataset"]
		isRealData = not hasattr(events, "genWeight")
		## Create histograms
		_hist_event_dict = (
			{"": None} if self.noHist else histogrammer(events, "ttcc2L2Nu")
		)
		if _hist_event_dict == None:
			_hist_event_dict[""]
		output = {
			"sumw": processor.defaultdict_accumulator(float),
			**_hist_event_dict,
		}
		if isRealData:
			output["sumw"] = len(events)
		else:
			output["sumw"] = ak.sum(events.genWeight)

		jets = events.Jet[:,:4] # use only 4 jets
		leptons = events.Lepton
			
		jet_drLep1 = jets.delta_r(leptons[:,0])
		jet_drLep2 = jets.delta_r(leptons[:,1])


		ch = events.Channel

		pruned_ev = {'Jet': jets, 'Lepton': leptons, 'Channel': ak.to_numpy(events.Channel), 'nbJet': ak.to_numpy(events.nbJet), 'ncJet': ak.to_numpy(events.ncJet)}
		pruned_ev['Jet_drLep1'] = ak.to_numpy(jet_drLep1)
		pruned_ev['Jet_drLep2'] = ak.to_numpy(jet_drLep2)

#		if self.isTTbar:
#			print('is ttbar sample')
#			pruned_ev.update({'isttbb': ak.to_numpy(events.isttbb), 'isttbj': ak.to_numpy(events.isttbj), 'isttcc': ak.to_numpy(events.isttcc), 'isttcj': ak.to_numpy(events.isttcj), 'isttother': ak.to_numpy(events.isttother)})
#
		#out_branch = ["events", "run", "luminosityBlock", "Channel", "trig_bit", "nbJet"]
		out_branch = ["Channel", "trig_bit", "nbJet", "ncJet"]

		for kin in ["pt", "eta", "phi", "mass"]:
			for obj in ["Jet", "Lepton"]:
				out_branch = np.append(out_branch, [f"{obj}_{kin}"])

		out_branch = np.append(
			out_branch,
			[
				"mll",
				"Jet_btag*",
				"Jet_DeepJet*",
				"Jet_drLep1",
				"Jet_drLep2",
			],
		)

		if self.isTTbar:
			print('is ttbar sample')
			pruned_ev.update({'isttbb': ak.to_numpy(events.isttbb), 'isttbj': ak.to_numpy(events.isttbj), 'isttcc': ak.to_numpy(events.isttcc), 'isttcj': ak.to_numpy(events.isttcj), 'isttother': ak.to_numpy(events.isttother)})
			out_branch = np.append(out_branch, ["isttbb", "isttbj", "isttcc", "isttcj", "isttother"])

		#os.system(f"mkdir -p {self.name}/{dataset}")
		os.system(f"mkdir -p {self.name}/ntuple")
		with uproot.recreate(
			#f"{self.name}/{dataset}/f{events.metadata['filename'].split('_')[-1].replace('.root','')}_{systematics[0]}_{int(events.metadata['entrystop']/self.chunksize)}.root"
			#f"{self.name}/ntuple/f{events.metadata['filename'].split('_')[-1].replace('.root','')}_{systematics[0]}_{int(events.metadata['entrystop']/self.chunksize)}.root"
			f"{self.name}/ntuple/test.root"
		) as fout:
			fout["Events"] = uproot_writeable(pruned_ev, include=out_branch)
		return {dataset: output}

	def postprocess(self, accumulator):
		return accumulator
