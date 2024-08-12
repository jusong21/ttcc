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
    #    events = missing_branch(events)
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

        req_nbjet = events.nbJets >= 2
        #njets = ak.num(events.Jet)
        jets = events.Jet[:,:4] # use only 4 jets
        jets_bsort = jets[ak.argsort(jets.btagDeepFlavB, ascending=False)]
        jets_bsort = jets_bsort[req_nbjet][:,:4]

        leptons = events.Lepton[req_nbjet]
        #dr_leptons = leptons[:,0].delta_r(leptons[:,1])
        met = events.MET[req_nbjet]


        jet_drLep1 = jets_bsort.delta_r(leptons[:,0])
        jet_drLep2 = jets_bsort.delta_r(leptons[:,1])

        #pruned_ev = {'Channel': ak.to_numpy(events.Channel), 'nJets': ak.to_numpy(njets), 'nbJets': ak.to_numpy(events.nbJets), 'nbJets_T': ak.to_numpy(events.nbJets_T), 'ncJets': ak.to_numpy(events.ncJets), 'ncJets_T': ak.to_numpy(events.ncJets_T)}
        pruned_ev = {'sortJet': jets_bsort, 'Jet': jets, 'Lepton': leptons, 'Channel': ak.to_numpy(events.Channel[req_nbjet]), 'nJets': ak.to_numpy(events.nJets[req_nbjet]), 'nbJets': ak.to_numpy(events.nbJets[req_nbjet]), 'ncJets': ak.to_numpy(events.ncJets[req_nbjet])}

        # leptons
#        pruned_ev['Lepton1_pt'] = ak.to_numpy(leptons.pt[:,0])
#        pruned_ev['Lepton1_eta'] = ak.to_numpy(leptons.eta[:,0])
#        pruned_ev['Lepton1_phi'] = ak.to_numpy(leptons.phi[:,0])
#        pruned_ev['Lepton1_mass'] = ak.to_numpy(leptons.mass[:,0])
#
#        pruned_ev['Lepton2_pt'] = ak.to_numpy(leptons.pt[:,1])
#        pruned_ev['Lepton2_eta'] = ak.to_numpy(leptons.eta[:,1])
#        pruned_ev['Lepton2_phi'] = ak.to_numpy(leptons.phi[:,1])
#        pruned_ev['Lepton2_mass'] = ak.to_numpy(leptons.mass[:,1])
#        
#        #pruned_ev['drLepton12'] = ak.to_numpy(dr_leptons)
        pruned_ev['massLepton12'] = ak.to_numpy(events.mll[req_nbjet])

        # MET
        pruned_ev['MET'] = ak.to_numpy(met.pt)
        pruned_ev['MET_phi'] = ak.to_numpy(met.phi)

        # jets
        pruned_ev['sortJet_drLep1'] = ak.to_numpy(jet_drLep1)
        pruned_ev['sortJet_drLep2'] = ak.to_numpy(jet_drLep2)

#        pruned_ev['Jet1_drLep1'] = ak.to_numpy(jet_drLep1[:,0])
#        pruned_ev['Jet2_drLep1'] = ak.to_numpy(jet_drLep1[:,1])
#        pruned_ev['Jet3_drLep1'] = ak.to_numpy(jet_drLep1[:,2])
#        pruned_ev['Jet4_drLep1'] = ak.to_numpy(jet_drLep1[:,3])
#
#        pruned_ev['Jet1_drLep2'] = ak.to_numpy(jet_drLep2[:,0])
#        pruned_ev['Jet2_drLep2'] = ak.to_numpy(jet_drLep2[:,1])
#        pruned_ev['Jet3_drLep2'] = ak.to_numpy(jet_drLep2[:,2])
#        pruned_ev['Jet4_drLep2'] = ak.to_numpy(jet_drLep2[:,3])
#
#        pruned_ev['Jet1_pt'] = ak.to_numpy(jets.pt[:,0])
#        pruned_ev['Jet1_eta'] = ak.to_numpy(jets.eta[:,0])
#        pruned_ev['Jet1_phi'] = ak.to_numpy(jets.phi[:,0])
#        pruned_ev['Jet1_mass'] = ak.to_numpy(jets.mass[:,0])
#        pruned_ev['Jet1_btagDeepFlavB'] = ak.to_numpy(jets.btagDeepFlavB[:,0])
#        pruned_ev['Jet1_btagDeepFlavCvB'] = ak.to_numpy(jets.btagDeepFlavCvB[:,0])
#        pruned_ev['Jet1_btagDeepFlavCvL'] = ak.to_numpy(jets.btagDeepFlavCvL[:,0])
#
#        pruned_ev['Jet2_pt'] = ak.to_numpy(jets.pt[:,1])
#        pruned_ev['Jet2_eta'] = ak.to_numpy(jets.eta[:,1])
#        pruned_ev['Jet2_phi'] = ak.to_numpy(jets.phi[:,1])
#        pruned_ev['Jet2_mass'] = ak.to_numpy(jets.mass[:,1])
#        pruned_ev['Jet2_btagDeepFlavB'] = ak.to_numpy(jets.btagDeepFlavB[:,1])
#        pruned_ev['Jet2_btagDeepFlavCvB'] = ak.to_numpy(jets.btagDeepFlavCvB[:,1])
#        pruned_ev['Jet2_btagDeepFlavCvL'] = ak.to_numpy(jets.btagDeepFlavCvL[:,1])
#
#        pruned_ev['Jet3_pt'] = ak.to_numpy(jets.pt[:,2])
#        pruned_ev['Jet3_eta'] = ak.to_numpy(jets.eta[:,2])
#        pruned_ev['Jet3_phi'] = ak.to_numpy(jets.phi[:,2])
#        pruned_ev['Jet3_mass'] = ak.to_numpy(jets.mass[:,2])
#        pruned_ev['Jet3_btagDeepFlavB'] = ak.to_numpy(jets.btagDeepFlavB[:,2])
#        pruned_ev['Jet3_btagDeepFlavCvB'] = ak.to_numpy(jets.btagDeepFlavCvB[:,2])
#        pruned_ev['Jet3_btagDeepFlavCvL'] = ak.to_numpy(jets.btagDeepFlavCvL[:,2])
#
#        pruned_ev['Jet4_pt'] = ak.to_numpy(jets.pt[:,3])
#        pruned_ev['Jet4_eta'] = ak.to_numpy(jets.eta[:,3])
#        pruned_ev['Jet4_phi'] = ak.to_numpy(jets.phi[:,3])
#        pruned_ev['Jet4_mass'] = ak.to_numpy(jets.mass[:,3])
#        pruned_ev['Jet4_btagDeepFlavB'] = ak.to_numpy(jets.btagDeepFlavB[:,3])
#        pruned_ev['Jet4_btagDeepFlavCvB'] = ak.to_numpy(jets.btagDeepFlavCvB[:,3])
#        pruned_ev['Jet4_btagDeepFlavCvL'] = ak.to_numpy(jets.btagDeepFlavCvL[:,3])

        # dijet
        pruned_ev['drJet12'] = ak.to_numpy(jets_bsort[:,0].delta_r(jets_bsort[:,1]))
        pruned_ev['drJet13'] = ak.to_numpy(jets_bsort[:,0].delta_r(jets_bsort[:,2]))
        pruned_ev['drJet14'] = ak.to_numpy(jets_bsort[:,0].delta_r(jets_bsort[:,3]))
        pruned_ev['drJet23'] = ak.to_numpy(jets_bsort[:,1].delta_r(jets_bsort[:,2]))
        pruned_ev['drJet24'] = ak.to_numpy(jets_bsort[:,1].delta_r(jets_bsort[:,3]))
        pruned_ev['drJet34'] = ak.to_numpy(jets_bsort[:,2].delta_r(jets_bsort[:,3]))

#        # btag sorting jets
#        pruned_ev['sortJet1_pt'] = ak.to_numpy(jets_bsort.pt[:,0])
#        pruned_ev['sortJet1_eta'] = ak.to_numpy(jets_bsort.eta[:,0])
#        pruned_ev['sortJet1_phi'] = ak.to_numpy(jets_bsort.phi[:,0])
#        pruned_ev['sortJet1_mass'] = ak.to_numpy(jets_bsort.mass[:,0])
#        pruned_ev['sortJet1_btagDeepFlavB'] = ak.to_numpy(jets_bsort.btagDeepFlavB[:,0])
#        pruned_ev['sortJet1_btagDeepFlavCvB'] = ak.to_numpy(jets_bsort.btagDeepFlavCvB[:,0])
#        pruned_ev['sortJet1_btagDeepFlavCvL'] = ak.to_numpy(jets_bsort.btagDeepFlavCvL[:,0])
#
#        pruned_ev['sortJet2_pt'] = ak.to_numpy(jets_bsort.pt[:,1])
#        pruned_ev['sortJet2_eta'] = ak.to_numpy(jets_bsort.eta[:,1])
#        pruned_ev['sortJet2_phi'] = ak.to_numpy(jets_bsort.phi[:,1])
#        pruned_ev['sortJet2_mass'] = ak.to_numpy(jets_bsort.mass[:,1])
#        pruned_ev['sortJet2_btagDeepFlavB'] = ak.to_numpy(jets_bsort.btagDeepFlavB[:,1])
#        pruned_ev['sortJet2_btagDeepFlavCvB'] = ak.to_numpy(jets_bsort.btagDeepFlavCvB[:,1])
#        pruned_ev['sortJet2_btagDeepFlavCvL'] = ak.to_numpy(jets_bsort.btagDeepFlavCvL[:,1])
#
#        pruned_ev['sortJet3_pt'] = ak.to_numpy(jets_bsort.pt[:,2])
#        pruned_ev['sortJet3_eta'] = ak.to_numpy(jets_bsort.eta[:,2])
#        pruned_ev['sortJet3_phi'] = ak.to_numpy(jets_bsort.phi[:,2])
#        pruned_ev['sortJet3_mass'] = ak.to_numpy(jets_bsort.mass[:,2])
#        pruned_ev['sortJet3_btagDeepFlavB'] = ak.to_numpy(jets_bsort.btagDeepFlavB[:,2])
#        pruned_ev['sortJet3_btagDeepFlavCvB'] = ak.to_numpy(jets_bsort.btagDeepFlavCvB[:,2])
#        pruned_ev['sortJet3_btagDeepFlavCvL'] = ak.to_numpy(jets_bsort.btagDeepFlavCvL[:,2])
#
#        pruned_ev['sortJet4_pt'] = ak.to_numpy(jets_bsort.pt[:,3])
#        pruned_ev['sortJet4_eta'] = ak.to_numpy(jets_bsort.eta[:,3])
#        pruned_ev['sortJet4_phi'] = ak.to_numpy(jets_bsort.phi[:,3])
#        pruned_ev['sortJet4_mass'] = ak.to_numpy(jets_bsort.mass[:,3])
#        pruned_ev['sortJet4_btagDeepFlavB'] = ak.to_numpy(jets_bsort.btagDeepFlavB[:,3])
#        pruned_ev['sortJet4_btagDeepFlavCvB'] = ak.to_numpy(jets_bsort.btagDeepFlavCvB[:,3])
#        pruned_ev['sortJet4_btagDeepFlavCvL'] = ak.to_numpy(jets_bsort.btagDeepFlavCvL[:,3])

        if isTTbar:
            pruned_ev.update({
                'isttbb': ak.to_numpy(events.isttbb[req_nbjet]), 
                'isttbj': ak.to_numpy(events.isttbj[req_nbjet]), 
                'isttcc': ak.to_numpy(events.isttcc[req_nbjet]), 
                'isttcj': ak.to_numpy(events.isttcj[req_nbjet]), 
                'isttother': ak.to_numpy(events.isttother[req_nbjet])
            })
#            #nbJetFromT = ak.num(events.nbJetsFromT)
#            for ibj in range(events.nbJetsFromT):
#                idx = ibj+1
#                pruned_ev[f"bJetFromT{idx}_pt"] = ak.to_numpy(events.bJetFromT.pt[:,ibj])
#                pruned_ev[f"bJetFromT{idx}_eta"] = ak.to_numpy(events.bJetFromT.eta[:,ibj])
#                pruned_ev[f"bJetFromT{idx}_phi"] = ak.to_numpy(events.bJetFromT.phi[:,ibj])
#                pruned_ev[f"bJetFromT{idx}_mass"] = ak.to_numpy(events.bJetFromT.mass[:,ibj])
#
#            for ibj in range(events.naddbJet):
#                idx = ibj+1
#                pruned_ev[f"addbJet{idx}_pt"] = ak.to_numpy(events.addbJet.pt[:,ibj])
#                pruned_ev[f"addbJet{idx}_eta"] = ak.to_numpy(events.addbJet.eta[:,ibj])
#                pruned_ev[f"addbJet{idx}_phi"] = ak.to_numpy(events.addbJet.phi[:,ibj])
#                pruned_ev[f"addbJet{idx}_mass"] = ak.to_numpy(events.addbJet.mass[:,ibj])
#
#            for icj in range(events.naddcJet):
#                idx = icj+1
#                pruned_ev[f"addcJet{idx}_pt"] = ak.to_numpy(events.addcJet.pt[:,icj])
#                pruned_ev[f"addcJet{idx}_eta"] = ak.to_numpy(events.addcJet.eta[:,icj])
#                pruned_ev[f"addcJet{idx}_phi"] = ak.to_numpy(events.addcJet.phi[:,icj])
#                pruned_ev[f"addcJet{idx}_mass"] = ak.to_numpy(events.addcJet.mass[:,icj])
#
#            for ilf in range(events.naddlfJet):
#                idx = ilf+1
#                pruned_ev[f"addlfJet{idx}_pt"] = ak.to_numpy(events.addlfJet.pt[:,ilf])
#                pruned_ev[f"addlfJet{idx}_eta"] = ak.to_numpy(events.addlfJet.eta[:,ilf])
#                pruned_ev[f"addlfJet{idx}_phi"] = ak.to_numpy(events.addlfJet.phi[:,ilf])
#                pruned_ev[f"addlfJet{idx}_mass"] = ak.to_numpy(events.addlfJet.mass[:,ilf])

#"""
#        if not isRealData:
#            print("is not RealData")
#            DeepJetC_totDown_weight, DeepJetC_totUp_weight = calc_tot_unc(events, "DeepJetCJet")
#            DeepJetB_totDown_weight, DeepJetB_totUp_weight = calc_tot_unc(events, "DeepJetBJet")
#
#            pruned_ev.update({
#               "weight": ak.to_numpy(events.weight),
#               "genweight_weight": ak.to_numpy(events.genweight.weight),
#               "puweight_weight": ak.to_numpy(events.puweight.weight),
#               "HLT_weight": ak.to_numpy(events.HLT.weight),
#               "mu_Reco_weight": ak.to_numpy(events.mu.Reco_weight),
#               "mu_ID_weight": ak.to_numpy(events.mu.ID_weight),
#               "mu_Iso_weight": ak.to_numpy(events.mu.Iso_weight),
#               "ele_Reco_weight": ak.to_numpy(events.ele.Reco_weight),
#               "ele_ID_weight": ak.to_numpy(events.ele.ID_weight),
#               "L1PreFiringWeight_Nom": ak.to_numpy(events.L1PreFiringWeight.Nom),
#               "DeepJetC_weight": ak.to_numpy(ak.prod(events.DeepJetCJet.weight, axis=-1)),
#               "DeepJetB_weight": ak.to_numpy(ak.prod(events.DeepJetBJet.weight, axis=-1)),
#
#               # up
#               "puweightUp_weight": ak.to_numpy(events.puweightUp.weight),
#               "HLTUp_weight": ak.to_numpy(events.HLTUp.weight),
#               "mu_RecoUp_weight": ak.to_numpy(events.mu.RecoUp_weight),
#               "mu_IDUp_weight": ak.to_numpy(events.mu.IDUp_weight),
#               "mu_IsoUp_weight": ak.to_numpy(events.mu.IsoUp_weight),
#               "ele_RecoUp_weight": ak.to_numpy(events.ele.RecoUp_weight),
#               "ele_IDUp_weight": ak.to_numpy(events.ele.IDUp_weight),
#               "DeepJetC_TotUp_weight": DeepJetC_totUp_weight,
#               "DeepJetB_TotUp_weight": DeepJetB_totUp_weight,
##               "DeepJetC_TotUp_weight": ak.to_numpy(DeepJetC_totUp_weight),
##               "DeepJetB_TotUp_weight": ak.to_numpy(DeepJetB_totUp_weight),
#               # need to check
#               "L1PreFiringWeight_Up": ak.to_numpy(events.L1PreFiringWeight.Dn),
#
#
#               # down
#               "puweightDown_weight": ak.to_numpy(events.puweightDown.weight),
#               "HLTDown_weight": ak.to_numpy(events.HLTDown.weight),
#               "mu_RecoDown_weight": ak.to_numpy(events.mu.RecoDown_weight),
#               "mu_IDDown_weight": ak.to_numpy(events.mu.IDDown_weight),
#               "mu_IsoDown_weight": ak.to_numpy(events.mu.IsoDown_weight),
#               "ele_RecoDown_weight": ak.to_numpy(events.ele.RecoDown_weight),
#               "ele_IDDown_weight": ak.to_numpy(events.ele.IDDown_weight),
#               "DeepJetC_TotDown_weight": DeepJetC_totDown_weight,
#               "DeepJetB_TotDown_weight": DeepJetB_totDown_weight,
##               "DeepJetC_TotDown_weight": ak.to_numpy(DeepJetC_totDown_weight),
##               "DeepJetB_TotDown_weight": ak.to_numpy(DeepJetB_totDown_weight),
#               # need to check
#               "L1PreFiringWeight_Down": ak.to_numpy(events.L1PreFiringWeight.Up),
#
#            })
#"""
        out_branch = list(pruned_ev.keys())
        for kin in ["pt", "eta", "phi", "mass"]:
            for obj in ["sortJet", "Jet", "Lepton"]:
                out_branch = np.append(out_branch, [f"{obj}_{kin}"])
        out_branch = np.append(out_branch, ["sortJet_btagDeepFlavB", "sortJet_btagDeepFlavCvB", "sortJet_btagDeepFlavCvL"])

        os.system(f"mkdir -p {self.name}/{dataset}")
        foutname = f"{self.name}/{dataset}/{events.metadata['filename'].split('_')[-1].replace('.root','')}_{int(events.metadata['entrystop']/self.chunksize)}.root"

        if isRealData: foutname = f"{self.name}/{dataset}/{events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(events.metadata['entrystop']/self.chunksize)}.root"

        with uproot.recreate( foutname) as fout:
            fout["Events"] = uproot_writeable(pruned_ev, include=out_branch)
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
