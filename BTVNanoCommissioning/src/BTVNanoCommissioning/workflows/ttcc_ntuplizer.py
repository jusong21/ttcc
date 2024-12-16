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
        print("dataset:", dataset)

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

        req_nbjet = events.nbJetsM >= 2
        #req_nbjet = events.nbJets >= 0
        #njets = ak.num(events.Jet)
        #jets = events.Jet[:,:4] # use only 4 jets
        jets = events.Jet
        jets_bsort = jets[ak.argsort(jets.btagDeepFlavB, ascending=False)]
        jets_bsort = jets_bsort[req_nbjet][:,:4]

        leptons = events.Lepton[req_nbjet]
        #dr_leptons = leptons[:,0].delta_r(leptons[:,1])
        met = events.MET[req_nbjet]
            
#        jet_drLep1 = ak.to_numpy(jets_bsort.delta_r(leptons[:,0]))
#        jet_drLep2 = ak.to_numpy(jets_bsort.delta_r(leptons[:,1]))
#        
#        lep_drJet1 = np.column_stack((jet_drLep1[:, 0], jet_drLep2[:, 0]))
#        lep_drJet2 = np.column_stack((jet_drLep1[:, 1], jet_drLep2[:, 1]))
#        lep_drJet3 = np.column_stack((jet_drLep1[:, 2], jet_drLep2[:, 2]))
#        lep_drJet4 = np.column_stack((jet_drLep1[:, 3], jet_drLep2[:, 3]))

        #lepton_drJet1 = transposed[:, 0]  # shape: (2,)
        #lepton_drJet2 = transposed[:, 1]  # shape: (2,)
        #lepton_drJet3 = transposed[:, 2]  # shape: (2,)
        #lepton_drJet4 = transposed[:, 3]  # shape: (2,)

        #lep_drJet1 = leptons.delta_r(jets_bsort[:,0])
        #lep_drJet2 = leptons.delta_r(jets_bsort[:,1])
        #lep_drJet3 = leptons.delta_r(jets_bsort[:,2])
        #lep_drJet4 = leptons.delta_r(jets_bsort[:,3])

        #pruned_ev = {'Channel': ak.to_numpy(events.Channel), 'nJets': ak.to_numpy(njets), 'nbJets': ak.to_numpy(events.nbJets), 'nbJets_T': ak.to_numpy(events.nbJets_T), 'ncJets': ak.to_numpy(events.ncJets), 'ncJets_T': ak.to_numpy(events.ncJets_T)}
        pruned_ev = {'sortJet': jets_bsort, 'Jet': jets[req_nbjet], 'Lepton': leptons, 'Channel': events.Channel[req_nbjet], 'nJets': events.nJets[req_nbjet], 'nbJetsL': events.nbJetsL[req_nbjet], "nbJetsM": events.nbJetsM[req_nbjet], 'nbJetsT': events.nbJetsT[req_nbjet], 'ncJetsL': events.ncJetsL[req_nbjet], 'ncJetsM': events.ncJetsM[req_nbjet], 'ncJetsT': events.ncJetsT[req_nbjet], }

        # leptons
#        #pruned_ev['drLepton12'] = ak.to_numpy(dr_leptons)
        pruned_ev['massLepton12'] = ak.to_numpy(events.mll[req_nbjet])
#        pruned_ev['Lepton_drJet1'] = ak.to_numpy(lep_drJet1)
#        pruned_ev['Lepton_drJet2'] = ak.to_numpy(lep_drJet2)
#        pruned_ev['Lepton_drJet3'] = ak.to_numpy(lep_drJet3)
#        pruned_ev['Lepton_drJet4'] = ak.to_numpy(lep_drJet4)

        # MET
        pruned_ev['MET'] = ak.to_numpy(met.pt)
        pruned_ev['MET_phi'] = ak.to_numpy(met.phi)

        # jets
#        pruned_ev['sortJet_drLep1'] = jet_drLep1
#        pruned_ev['sortJet_drLep2'] = jet_drLep2

        # dijet
#        pruned_ev['drJet12'] = ak.to_numpy(jets_bsort[:,0].delta_r(jets_bsort[:,1]))
#        pruned_ev['drJet13'] = ak.to_numpy(jets_bsort[:,0].delta_r(jets_bsort[:,2]))
#        pruned_ev['drJet14'] = ak.to_numpy(jets_bsort[:,0].delta_r(jets_bsort[:,3]))
#        pruned_ev['drJet23'] = ak.to_numpy(jets_bsort[:,1].delta_r(jets_bsort[:,2]))
#        pruned_ev['drJet24'] = ak.to_numpy(jets_bsort[:,1].delta_r(jets_bsort[:,3]))
#        pruned_ev['drJet34'] = ak.to_numpy(jets_bsort[:,2].delta_r(jets_bsort[:,3]))

#        # btag sorting jets

        out_branch = ["events", "run", "luminosityBlock"]
        if isTTbar:
            pruned_ev.update({
                "ttbar_category": events.ttbar.category[req_nbjet],
                "genTtbarId": events.genTtbarId[req_nbjet],
            #    "nGenJets": ak.to_numpy(events.nGenJets[req_nbjet]),
            })

        if not isRealData:
            print("is not RealData")
#            DeepJetC_totDown_weight, DeepJetC_totUp_weight = calc_tot_unc(events, "DeepJetCJet")
#            DeepJetB_totDown_weight, DeepJetB_totUp_weight = calc_tot_unc(events, "DeepJetBJet")

            pruned_ev.update({
               "weight": events.weight[req_nbjet],
               "genweight_weight": events.genweight.weight[req_nbjet],
               "puweight_weight": events.puweight.weight[req_nbjet],
               "HLT_weight": events.HLT.weight[req_nbjet],
               "mu_Reco_weight": events.mu.Reco_weight[req_nbjet],
               "mu_ID_weight": events.mu.ID_weight[req_nbjet],
               "mu_Iso_weight": events.mu.Iso_weight[req_nbjet],
               "ele_Reco_weight": events.ele.Reco_weight[req_nbjet],
               "ele_ID_weight": events.ele.ID_weight[req_nbjet],
               "L1PreFiring_weight": events.L1PreFiringWeight.Nom[req_nbjet],
               "PNet_weight": events.PNet.weight[req_nbjet],
#               "DeepJetC_weight": ak.to_numpy(ak.prod(events.DeepJetCJet.weight[req_nbjet], axis=-1)),
#               "DeepJetB_weight": ak.to_numpy(ak.prod(events.DeepJetBJet.weight[req_nbjet], axis=-1)),

               # up
               "puweightUp_weight": events.puweightUp.weight[req_nbjet],
               "HLTUp_weight": events.HLTUp.weight[req_nbjet],
               "mu_RecoUp_weight": events.mu.RecoUp_weight[req_nbjet],
               "mu_IDUp_weight": events.mu.IDUp_weight[req_nbjet],
               "mu_IsoUp_weight": events.mu.IsoUp_weight[req_nbjet],
               "ele_RecoUp_weight": events.ele.RecoUp_weight[req_nbjet],
               "ele_IDUp_weight": events.ele.IDUp_weight[req_nbjet],
#               "DeepJetC_TotUp_weight": DeepJetC_totUp_weight[req_nbjet],
#               "DeepJetB_TotUp_weight": DeepJetB_totUp_weight[req_nbjet],
               # need to check
               "L1PreFiringUp_weight": events.L1PreFiringWeight.Dn[req_nbjet],
#
#
               # down
               "puweightDown_weight": events.puweightDown.weight[req_nbjet],
               "HLTDown_weight": events.HLTDown.weight[req_nbjet],
               "mu_RecoDown_weight": events.mu.RecoDown_weight[req_nbjet],
               "mu_IDDown_weight": events.mu.IDDown_weight[req_nbjet],
               "mu_IsoDown_weight": events.mu.IsoDown_weight[req_nbjet],
               "ele_RecoDown_weight": events.ele.RecoDown_weight[req_nbjet],
               "ele_IDDown_weight": events.ele.IDDown_weight[req_nbjet],
#               "DeepJetC_TotDown_weight": DeepJetC_totDown_weight[req_nbjet],
#               "DeepJetB_TotDown_weight": DeepJetB_totDown_weight[req_nbjet],
               # need to check
               "L1PreFiringDown_weight": events.L1PreFiringWeight.Up[req_nbjet],
#
            })

#            for kin in ["pt", "eta"]:
#                for obj in ["bJetFromT", "bJetFromW", "addbJet", "addcJet", "addlfJet"]:
#                    out_branch = np.append(out_branch, [f"{obj}_{kin}"])
#"""
        out_branch = np.append(out_branch, list(pruned_ev.keys()))
        #out_branch = list(pruned_ev.keys())
        for kin in ["pt", "eta", "phi", "mass"]:
            for obj in ["sortJet", "Jet", "Lepton"]:
                out_branch = np.append(out_branch, [f"{obj}_{kin}"])
        out_branch = np.append(out_branch, ["sortJet_btagDeepFlavB", "sortJet_btagDeepFlavCvB", "sortJet_btagDeepFlavCvL"])

        os.system(f"mkdir -p {self.name}/{dataset}")
        foutname = f"{self.name}/{dataset}/{events.metadata['filename'].split('_')[-1].replace('.root','')}_{int(events.metadata['entrystop']/self.chunksize)}.root"

        if isRealData: foutname = f"{self.name}/{dataset}/{events.metadata['filename'].split('/')[-1].replace('.root','')}_{int(events.metadata['entrystop']/self.chunksize)}.root"

        if len(pruned_ev["nJets"]) == 0:
            print(f"Skipping writing {dataset} {foutname} as all branches are empty.")
            return {dataset: output}

        with uproot.recreate( foutname) as fout:
            fout["Events"] = uproot_writeable(pruned_ev, include=out_branch)
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
