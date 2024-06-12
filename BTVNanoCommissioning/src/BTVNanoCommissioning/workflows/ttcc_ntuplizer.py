import collections, awkward as ak, numpy as np
import os
import uproot
from coffea import processor
from coffea.analysis_tools import Weights

# user helper function
from BTVNanoCommissioning.helpers.func import (
	flatten,
	update,
	uproot_writeable,
	dump_lumi,
)
from BTVNanoCommissioning.utils.histogrammer import histogrammer

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
		#self.lumiMask = load_lumi(self._campaign)
		self.chunksize = chunksize
		## Load corrections
		#self.SF_map = load_SF(self._campaign)
		self.isTTbar = isTTbar

	@property
	def accumulator(self):
		return self._accumulator

	## Apply corrections on momentum/mass on MET, Jet, Muon
	def process(self, events):
		print("**************************************")
		print("This is ", self._campaign, "ttcc dilepton channel ntuplizer")
		print("**************************************")
		dataset = events.metadata["dataset"]
	#	events = missing_branch(events)
		return processor.accumulate(
			self.process_shift(events)
		)

	def process_shift(self, events):
		dataset = events.metadata["dataset"]
		isRealData = not hasattr(events, "genweight")
		isTTbar = 'TTTo' in dataset

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
			output["sumw"] = ak.sum(events.genweight)
		njets = ak.num(events.Jet)
		jets = events.Jet[:,:4] # use only 4 jets
		jets_bsort = jets[ak.argsort(jets.btagDeepFlavB, ascending=False)]

		leptons = events.Lepton
		#dr_leptons = leptons[:,0].delta_r(leptons[:,1])
		met = events.MET


		jet_drLep1 = jets.delta_r(leptons[:,0])
		jet_drLep2 = jets.delta_r(leptons[:,1])

		pruned_ev = {'Channel': ak.to_numpy(events.Channel), 'nJets': ak.to_numpy(njets), 'nbJets': ak.to_numpy(events.nbJet)}

		# leptons
		pruned_ev['Lepton1_pt'] = ak.to_numpy(leptons.pt[:,0])
		pruned_ev['Lepton1_eta'] = ak.to_numpy(leptons.eta[:,0])
		pruned_ev['Lepton1_phi'] = ak.to_numpy(leptons.phi[:,0])
		pruned_ev['Lepton1_mass'] = ak.to_numpy(leptons.mass[:,0])

		pruned_ev['Lepton2_pt'] = ak.to_numpy(leptons.pt[:,1])
		pruned_ev['Lepton2_eta'] = ak.to_numpy(leptons.eta[:,1])
		pruned_ev['Lepton2_phi'] = ak.to_numpy(leptons.phi[:,1])
		pruned_ev['Lepton2_mass'] = ak.to_numpy(leptons.mass[:,1])
		
		#pruned_ev['drLepton12'] = ak.to_numpy(dr_leptons)
		pruned_ev['massLepton12'] = ak.to_numpy(events.mll)

		# MET
		pruned_ev['MET'] = ak.to_numpy(met.pt)
		pruned_ev['MET_phi'] = ak.to_numpy(met.phi)

		# jets
		pruned_ev['Jet_drLep1'] = ak.to_numpy(jet_drLep1)
		pruned_ev['Jet_drLep2'] = ak.to_numpy(jet_drLep2)

		pruned_ev['Jet1_drLep1'] = ak.to_numpy(jet_drLep1[:,0])
		pruned_ev['Jet2_drLep1'] = ak.to_numpy(jet_drLep1[:,1])
		pruned_ev['Jet3_drLep1'] = ak.to_numpy(jet_drLep1[:,2])
		pruned_ev['Jet4_drLep1'] = ak.to_numpy(jet_drLep1[:,3])

		pruned_ev['Jet1_drLep2'] = ak.to_numpy(jet_drLep2[:,0])
		pruned_ev['Jet2_drLep2'] = ak.to_numpy(jet_drLep2[:,1])
		pruned_ev['Jet3_drLep2'] = ak.to_numpy(jet_drLep2[:,2])
		pruned_ev['Jet4_drLep2'] = ak.to_numpy(jet_drLep2[:,3])

		pruned_ev['Jet1_pt'] = ak.to_numpy(jets.pt[:,0])
		pruned_ev['Jet1_eta'] = ak.to_numpy(jets.eta[:,0])
		pruned_ev['Jet1_phi'] = ak.to_numpy(jets.phi[:,0])
		pruned_ev['Jet1_mass'] = ak.to_numpy(jets.mass[:,0])
		pruned_ev['Jet1_btagDeepFlavB'] = ak.to_numpy(jets.btagDeepFlavB[:,0])
		pruned_ev['Jet1_btagDeepFlavCvB'] = ak.to_numpy(jets.btagDeepFlavCvB[:,0])
		pruned_ev['Jet1_btagDeepFlavCvL'] = ak.to_numpy(jets.btagDeepFlavCvL[:,0])

		pruned_ev['Jet2_pt'] = ak.to_numpy(jets.pt[:,1])
		pruned_ev['Jet2_eta'] = ak.to_numpy(jets.eta[:,1])
		pruned_ev['Jet2_phi'] = ak.to_numpy(jets.phi[:,1])
		pruned_ev['Jet2_mass'] = ak.to_numpy(jets.mass[:,1])
		pruned_ev['Jet2_btagDeepFlavB'] = ak.to_numpy(jets.btagDeepFlavB[:,1])
		pruned_ev['Jet2_btagDeepFlavCvB'] = ak.to_numpy(jets.btagDeepFlavCvB[:,1])
		pruned_ev['Jet2_btagDeepFlavCvL'] = ak.to_numpy(jets.btagDeepFlavCvL[:,1])

		pruned_ev['Jet3_pt'] = ak.to_numpy(jets.pt[:,2])
		pruned_ev['Jet3_eta'] = ak.to_numpy(jets.eta[:,2])
		pruned_ev['Jet3_phi'] = ak.to_numpy(jets.phi[:,2])
		pruned_ev['Jet3_mass'] = ak.to_numpy(jets.mass[:,2])
		pruned_ev['Jet3_btagDeepFlavB'] = ak.to_numpy(jets.btagDeepFlavB[:,2])
		pruned_ev['Jet3_btagDeepFlavCvB'] = ak.to_numpy(jets.btagDeepFlavCvB[:,2])
		pruned_ev['Jet3_btagDeepFlavCvL'] = ak.to_numpy(jets.btagDeepFlavCvL[:,2])

		pruned_ev['Jet4_pt'] = ak.to_numpy(jets.pt[:,3])
		pruned_ev['Jet4_eta'] = ak.to_numpy(jets.eta[:,3])
		pruned_ev['Jet4_phi'] = ak.to_numpy(jets.phi[:,3])
		pruned_ev['Jet4_mass'] = ak.to_numpy(jets.mass[:,3])
		pruned_ev['Jet4_btagDeepFlavB'] = ak.to_numpy(jets.btagDeepFlavB[:,3])
		pruned_ev['Jet4_btagDeepFlavCvB'] = ak.to_numpy(jets.btagDeepFlavCvB[:,3])
		pruned_ev['Jet4_btagDeepFlavCvL'] = ak.to_numpy(jets.btagDeepFlavCvL[:,3])

		# dijet
		pruned_ev['drJet12'] = ak.to_numpy(jets[:,0].delta_r(jets[:,1]))
		pruned_ev['drJet13'] = ak.to_numpy(jets[:,0].delta_r(jets[:,2]))
		pruned_ev['drJet14'] = ak.to_numpy(jets[:,0].delta_r(jets[:,3]))
		pruned_ev['drJet23'] = ak.to_numpy(jets[:,1].delta_r(jets[:,2]))
		pruned_ev['drJet24'] = ak.to_numpy(jets[:,1].delta_r(jets[:,3]))
		pruned_ev['drJet34'] = ak.to_numpy(jets[:,2].delta_r(jets[:,3]))

		# btag sorting jets
		pruned_ev['sortJet1_pt'] = ak.to_numpy(jets_bsort.pt[:,0])
		pruned_ev['sortJet1_eta'] = ak.to_numpy(jets_bsort.eta[:,0])
		pruned_ev['sortJet1_phi'] = ak.to_numpy(jets_bsort.phi[:,0])
		pruned_ev['sortJet1_mass'] = ak.to_numpy(jets_bsort.mass[:,0])
		pruned_ev['sortJet1_btagDeepFlavB'] = ak.to_numpy(jets_bsort.btagDeepFlavB[:,0])
		pruned_ev['sortJet1_btagDeepFlavCvB'] = ak.to_numpy(jets_bsort.btagDeepFlavCvB[:,0])
		pruned_ev['sortJet1_btagDeepFlavCvL'] = ak.to_numpy(jets_bsort.btagDeepFlavCvL[:,0])

		pruned_ev['sortJet2_pt'] = ak.to_numpy(jets_bsort.pt[:,1])
		pruned_ev['sortJet2_eta'] = ak.to_numpy(jets_bsort.eta[:,1])
		pruned_ev['sortJet2_phi'] = ak.to_numpy(jets_bsort.phi[:,1])
		pruned_ev['sortJet2_mass'] = ak.to_numpy(jets_bsort.mass[:,1])
		pruned_ev['sortJet2_btagDeepFlavB'] = ak.to_numpy(jets_bsort.btagDeepFlavB[:,1])
		pruned_ev['sortJet2_btagDeepFlavCvB'] = ak.to_numpy(jets_bsort.btagDeepFlavCvB[:,1])
		pruned_ev['sortJet2_btagDeepFlavCvL'] = ak.to_numpy(jets_bsort.btagDeepFlavCvL[:,1])

		pruned_ev['sortJet3_pt'] = ak.to_numpy(jets_bsort.pt[:,2])
		pruned_ev['sortJet3_eta'] = ak.to_numpy(jets_bsort.eta[:,2])
		pruned_ev['sortJet3_phi'] = ak.to_numpy(jets_bsort.phi[:,2])
		pruned_ev['sortJet3_mass'] = ak.to_numpy(jets_bsort.mass[:,2])
		pruned_ev['sortJet3_btagDeepFlavB'] = ak.to_numpy(jets_bsort.btagDeepFlavB[:,2])
		pruned_ev['sortJet3_btagDeepFlavCvB'] = ak.to_numpy(jets_bsort.btagDeepFlavCvB[:,2])
		pruned_ev['sortJet3_btagDeepFlavCvL'] = ak.to_numpy(jets_bsort.btagDeepFlavCvL[:,2])

		pruned_ev['sortJet4_pt'] = ak.to_numpy(jets_bsort.pt[:,3])
		pruned_ev['sortJet4_eta'] = ak.to_numpy(jets_bsort.eta[:,3])
		pruned_ev['sortJet4_phi'] = ak.to_numpy(jets_bsort.phi[:,3])
		pruned_ev['sortJet4_mass'] = ak.to_numpy(jets_bsort.mass[:,3])
		pruned_ev['sortJet4_btagDeepFlavB'] = ak.to_numpy(jets_bsort.btagDeepFlavB[:,3])
		pruned_ev['sortJet4_btagDeepFlavCvB'] = ak.to_numpy(jets_bsort.btagDeepFlavCvB[:,3])
		pruned_ev['sortJet4_btagDeepFlavCvL'] = ak.to_numpy(jets_bsort.btagDeepFlavCvL[:,3])

#		if isTTbar:
#			pruned_ev.update({'isttbb': ak.to_numpy(events.isttbb), 'isttbj': ak.to_numpy(events.isttbj), 'isttcc': ak.to_numpy(events.isttcc), 'isttcj': ak.to_numpy(events.isttcj), 'isttother': ak.to_numpy(events.isttother)})

		if not isRealData:
			pruned_ev.update({
			   'weight': ak.to_numpy(events.weight),
			   'genweight_weight': ak.to_numpy(events.genweight.weight),
			   'puweight_weight': ak.to_numpy(events.puweight.weight),
			   'HLT_weight': ak.to_numpy(events.HLT.weight),
			   'mu_Reco_weight': ak.to_numpy(events.mu.Reco_weight),
			   'mu_ID_weight': ak.to_numpy(events.mu.ID_weight),
			   'mu_Iso_weight': ak.to_numpy(events.mu.Iso_weight),
			   'ele_Reco_weight': ak.to_numpy(events.ele.Reco_weight),
			   'ele_ID_weight': ak.to_numpy(events.ele.ID_weight),
			   'DeepJetC_weight': ak.to_numpy(events.DeepJetC.weight)
			})
		
		out_branch = list(pruned_ev.keys())

		os.system(f"mkdir -p {self.name}/{dataset}")
		foutname = f"{self.name}/{dataset}/{events.metadata['filename'].split('_')[-1].replace('.root','')}_{int(events.metadata['entrystop']/self.chunksize)}.root"

		if isRealData: foutname = f"{self.name}/{dataset}/{events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(events.metadata['entrystop']/self.chunksize)}.root"

		with uproot.recreate( foutname) as fout:
			fout["Events"] = uproot_writeable(pruned_ev, include=out_branch)
		return {dataset: output}

	def postprocess(self, accumulator):
		return accumulator
