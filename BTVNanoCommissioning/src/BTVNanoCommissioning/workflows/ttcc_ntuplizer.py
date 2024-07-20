import collections, awkward as ak, numpy as np
import os
import uproot
from coffea import processor
from coffea.analysis_tools import Weights
import math
from BTVNanoCommissioning.helpers.ttcc2L2Nu_helper import calc_tot_unc

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
		isRealData = not hasattr(events, "weight")
		#isTTbar = 'TTTo' in dataset
		isTTbar = ("TTTo" in dataset) or ("TTbb" in dataset)

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
			output["sumw"] = ak.sum(events.weight)
		njets = ak.num(events.Jet)
		jets = events.Jet[:,:4] # use only 4 jets
		jets_bsort = jets[ak.argsort(jets.btagDeepFlavB, ascending=False)]

		leptons = events.Lepton
		#dr_leptons = leptons[:,0].delta_r(leptons[:,1])
		met = events.MET


		jet_drLep1 = jets.delta_r(leptons[:,0])
		jet_drLep2 = jets.delta_r(leptons[:,1])

		#pruned_ev = {'Channel': ak.to_numpy(events.Channel), 'nJets': ak.to_numpy(njets), 'nbJets': ak.to_numpy(events.nbJets), 'nbJets_T': ak.to_numpy(events.nbJets_T), 'ncJets': ak.to_numpy(events.ncJets), 'ncJets_T': ak.to_numpy(events.ncJets_T)}
		pruned_ev = {'Channel': ak.to_numpy(events.Channel), 'nJets': ak.to_numpy(njets), 'nbJets': ak.to_numpy(events.nbJets), 'ncJets': ak.to_numpy(events.ncJets)}

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

		if isTTbar:
			pruned_ev.update({
				'isttbb': ak.to_numpy(events.isttbb), 
				'isttbj': ak.to_numpy(events.isttbj), 
				'isttcc': ak.to_numpy(events.isttcc), 
				'isttcj': ak.to_numpy(events.isttcj), 
				'isttother': ak.to_numpy(events.isttother)
			})
#			#nbJetFromT = ak.num(events.nbJetsFromT)
#			for ibj in range(events.nbJetsFromT):
#				idx = ibj+1
#				pruned_ev[f"bJetFromT{idx}_pt"] = ak.to_numpy(events.bJetFromT.pt[:,ibj])
#				pruned_ev[f"bJetFromT{idx}_eta"] = ak.to_numpy(events.bJetFromT.eta[:,ibj])
#				pruned_ev[f"bJetFromT{idx}_phi"] = ak.to_numpy(events.bJetFromT.phi[:,ibj])
#				pruned_ev[f"bJetFromT{idx}_mass"] = ak.to_numpy(events.bJetFromT.mass[:,ibj])
#
#			for ibj in range(events.naddbJet):
#				idx = ibj+1
#				pruned_ev[f"addbJet{idx}_pt"] = ak.to_numpy(events.addbJet.pt[:,ibj])
#				pruned_ev[f"addbJet{idx}_eta"] = ak.to_numpy(events.addbJet.eta[:,ibj])
#				pruned_ev[f"addbJet{idx}_phi"] = ak.to_numpy(events.addbJet.phi[:,ibj])
#				pruned_ev[f"addbJet{idx}_mass"] = ak.to_numpy(events.addbJet.mass[:,ibj])
#
#			for icj in range(events.naddcJet):
#				idx = icj+1
#				pruned_ev[f"addcJet{idx}_pt"] = ak.to_numpy(events.addcJet.pt[:,icj])
#				pruned_ev[f"addcJet{idx}_eta"] = ak.to_numpy(events.addcJet.eta[:,icj])
#				pruned_ev[f"addcJet{idx}_phi"] = ak.to_numpy(events.addcJet.phi[:,icj])
#				pruned_ev[f"addcJet{idx}_mass"] = ak.to_numpy(events.addcJet.mass[:,icj])
#
#			for ilf in range(events.naddlfJet):
#				idx = ilf+1
#				pruned_ev[f"addlfJet{idx}_pt"] = ak.to_numpy(events.addlfJet.pt[:,ilf])
#				pruned_ev[f"addlfJet{idx}_eta"] = ak.to_numpy(events.addlfJet.eta[:,ilf])
#				pruned_ev[f"addlfJet{idx}_phi"] = ak.to_numpy(events.addlfJet.phi[:,ilf])
#				pruned_ev[f"addlfJet{idx}_mass"] = ak.to_numpy(events.addlfJet.mass[:,ilf])


		if not isRealData:
			print("is not RealData")
			DeepJetC_totDown_weight, DeepJetC_totUp_weight = calc_tot_unc(events, "DeepJetCJet")
			DeepJetB_totDown_weight, DeepJetB_totUp_weight = calc_tot_unc(events, "DeepJetBJet")

			pruned_ev.update({
			   "weight": ak.to_numpy(events.weight),
			   "genweight_weight": ak.to_numpy(events.genweight.weight),
			   "puweight_weight": ak.to_numpy(events.puweight.weight),
			   "HLT_weight": ak.to_numpy(events.HLT.weight),
			   "mu_Reco_weight": ak.to_numpy(events.mu.Reco_weight),
			   "mu_ID_weight": ak.to_numpy(events.mu.ID_weight),
			   "mu_Iso_weight": ak.to_numpy(events.mu.Iso_weight),
			   "ele_Reco_weight": ak.to_numpy(events.ele.Reco_weight),
			   "ele_ID_weight": ak.to_numpy(events.ele.ID_weight),
			   "L1PreFiringWeight_Nom": ak.to_numpy(events.L1PreFiringWeight.Nom),
			   "DeepJetC_weight": ak.to_numpy(events.DeepJetCJet.weight),
			   "DeepJetB_weight": ak.to_numpy(events.DeepJetBJet.weight),

			   # up
			   "puweightUp_weight": ak.to_numpy(events.puweightUp.weight),
			   "HLTUp_weight": ak.to_numpy(events.HLTUp.weight),
			   "mu_RecoUp_weight": ak.to_numpy(events.mu.RecoUp_weight),
			   "mu_IDUp_weight": ak.to_numpy(events.mu.IDUp_weight),
			   "mu_IsoUp_weight": ak.to_numpy(events.mu.IsoUp_weight),
			   "ele_RecoUp_weight": ak.to_numpy(events.ele.RecoUp_weight),
			   "ele_IDUp_weight": ak.to_numpy(events.ele.IDUp_weight),
			   "DeepJetC_TotUp_weight": DeepJetC_totUp_weight,
			   "DeepJetB_TotUp_weight": DeepJetB_totUp_weight,
#			   "DeepJetC_TotUp_weight": ak.to_numpy(DeepJetC_totUp_weight),
#			   "DeepJetB_TotUp_weight": ak.to_numpy(DeepJetB_totUp_weight),
			   # need to check
			   "L1PreFiringWeight_Up": ak.to_numpy(events.L1PreFiringWeight.Dn),

#			   "DeepJetC_ExtrapUp_weight": ak.to_numpy(events.DeepJetC.ExtrapUp_weight),
#			   "DeepJetC_InterpUp_weight": ak.to_numpy(events.DeepJetC.InterpUp_weight),
#			   "DeepJetC_LHEScaleWeight_muFUp_weight": ak.to_numpy(events.DeepJetC.LHEScaleWeight_muFUp_weight),
#			   "DeepJetC_LHEScaleWeight_muRUp_weight": ak.to_numpy(events.DeepJetC.LHEScaleWeight_muRUp_weight),
#			   "DeepJetC_PSWeightFSRUp_weight": ak.to_numpy(events.DeepJetC.PSWeightFSRUp_weight),
#			   "DeepJetC_PSWeightISRUp_weight": ak.to_numpy(events.DeepJetC.PSWeightISRUp_weight),
#			   "DeepJetC_PUWeightUp_weight": ak.to_numpy(events.DeepJetC.PUWeightUp_weight),
#			   "DeepJetC_StatUp_weight": ak.to_numpy(events.DeepJetC.StatUp_weight),
#			   "DeepJetC_XSec_BRUnc_DYJets_bUp_weight": ak.to_numpy(events.DeepJetC.XSec_BRUnc_DYJets_bUp_weight),
#			   "DeepJetC_XSec_BRUnc_DYJets_cUp_weight": ak.to_numpy(events.DeepJetC.XSec_BRUnc_DYJets_cUp_weight),
#			   "DeepJetC_XSec_BRUnc_WJets_cUp_weight": ak.to_numpy(events.DeepJetC.XSec_BRUnc_WJets_cUp_weight),
#			   "DeepJetC_jerUp_weight": ak.to_numpy(events.DeepJetC.jerUp_weight),
#			   "DeepJetC_jesTotalUp_weight": ak.to_numpy(events.DeepJetC.jesTotalUp_weight),
#
#			   "DeepJetB_hfUp_weight": ak.to_numpy(events.DeepJetB.hfUp_weight),
#			   "DeepJetB_lfUp_weight": ak.to_numpy(events.DeepJetB.lfUp_weight),
#			   "DeepJetB_cferr1Up_weight": ak.to_numpy(events.DeepJetB.cferr1Up_weight),
#			   "DeepJetB_cferr2Up_weight": ak.to_numpy(events.DeepJetB.cferr2Up_weight),
#			   "DeepJetB_hfstats1Up_weight": ak.to_numpy(events.DeepJetB.hfstats1Up_weight),
#			   "DeepJetB_hfstats2Up_weight": ak.to_numpy(events.DeepJetB.hfstats2Up_weight),
#			   "DeepJetB_lfstats1Up_weight": ak.to_numpy(events.DeepJetB.lfstats1Up_weight),
#			   "DeepJetB_lfstats2Up_weight": ak.to_numpy(events.DeepJetB.lfstats2Up_weight),

			   # down
			   "puweightDown_weight": ak.to_numpy(events.puweightDown.weight),
			   "HLTDown_weight": ak.to_numpy(events.HLTDown.weight),
			   "mu_RecoDown_weight": ak.to_numpy(events.mu.RecoDown_weight),
			   "mu_IDDown_weight": ak.to_numpy(events.mu.IDDown_weight),
			   "mu_IsoDown_weight": ak.to_numpy(events.mu.IsoDown_weight),
			   "ele_RecoDown_weight": ak.to_numpy(events.ele.RecoDown_weight),
			   "ele_IDDown_weight": ak.to_numpy(events.ele.IDDown_weight),
			   "DeepJetC_TotDown_weight": DeepJetC_totDown_weight,
			   "DeepJetB_TotDown_weight": DeepJetB_totDown_weight,
#			   "DeepJetC_TotDown_weight": ak.to_numpy(DeepJetC_totDown_weight),
#			   "DeepJetB_TotDown_weight": ak.to_numpy(DeepJetB_totDown_weight),
			   # need to check
			   "L1PreFiringWeight_Down": ak.to_numpy(events.L1PreFiringWeight.Up),

#			   "DeepJetC_ExtrapDown_weight": ak.to_numpy(events.DeepJetC.ExtrapDown_weight),
#			   "DeepJetC_InterpDown_weight": ak.to_numpy(events.DeepJetC.InterpDown_weight),
#			   "DeepJetC_LHEScaleWeight_muFDown_weight": ak.to_numpy(events.DeepJetC.LHEScaleWeight_muFDown_weight),
#			   "DeepJetC_LHEScaleWeight_muRDown_weight": ak.to_numpy(events.DeepJetC.LHEScaleWeight_muRDown_weight),
#			   "DeepJetC_PSWeightFSRDown_weight": ak.to_numpy(events.DeepJetC.PSWeightFSRDown_weight),
#			   "DeepJetC_PSWeightISRDown_weight": ak.to_numpy(events.DeepJetC.PSWeightISRDown_weight),
#			   "DeepJetC_PUWeightDown_weight": ak.to_numpy(events.DeepJetC.PUWeightDown_weight),
#			   "DeepJetC_StatDown_weight": ak.to_numpy(events.DeepJetC.StatDown_weight),
#			   "DeepJetC_XSec_BRUnc_DYJets_bDown_weight": ak.to_numpy(events.DeepJetC.XSec_BRUnc_DYJets_bDown_weight),
#			   "DeepJetC_XSec_BRUnc_DYJets_cDown_weight": ak.to_numpy(events.DeepJetC.XSec_BRUnc_DYJets_cDown_weight),
#			   "DeepJetC_XSec_BRUnc_WJets_cDown_weight": ak.to_numpy(events.DeepJetC.XSec_BRUnc_WJets_cDown_weight),
#			   "DeepJetC_jerDown_weight": ak.to_numpy(events.DeepJetC.jerDown_weight),
#			   "DeepJetC_jesTotalDown_weight": ak.to_numpy(events.DeepJetC.jesTotalDown_weight),
#
#			   "DeepJetB_hfDown_weight": ak.to_numpy(events.DeepJetB.hfDown_weight),
#			   "DeepJetB_lfDown_weight": ak.to_numpy(events.DeepJetB.lfDown_weight),
#			   "DeepJetB_cferr1Down_weight": ak.to_numpy(events.DeepJetB.cferr1Down_weight),
#			   "DeepJetB_cferr2Down_weight": ak.to_numpy(events.DeepJetB.cferr2Down_weight),
#			   "DeepJetB_hfstats1Down_weight": ak.to_numpy(events.DeepJetB.hfstats1Down_weight),
#			   "DeepJetB_hfstats2Down_weight": ak.to_numpy(events.DeepJetB.hfstats2Down_weight),
#			   "DeepJetB_lfstats1Down_weight": ak.to_numpy(events.DeepJetB.lfstats1Down_weight),
#			   "DeepJetB_lfstats2Down_weight": ak.to_numpy(events.DeepJetB.lfstats2Down_weight),
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
