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
        #req_nbjet = events.nbJets >= 0
        #njets = ak.num(events.Jet)
        #jets = events.Jet[:,:4] # use only 4 jets
        jets = events.Jet
        jets_bsort = jets[ak.argsort(jets.btagDeepFlavB, ascending=False)]
        jets_bsort = jets_bsort[req_nbjet][:,:4]

        leptons = events.Lepton[req_nbjet]
        #dr_leptons = leptons[:,0].delta_r(leptons[:,1])
        met = events.MET[req_nbjet]

        # genJets category
        genbjet = events.Jet[events.Jet.hadronFlavour==5]
        gencjet = events.Jet[events.Jet.hadronFlavour==4]
        genlfjet = events.Jet[events.Jet.hadronFlavour==0]
        
        ngenbjets = ak.num(genbjet)
        ngencjets = ak.num(gencjet)
        ngenlfjets = ak.num(genlfjet)

        ttbar_criteria = {
            'ttbb': ngenbjets >= 4,
            'ttbj': ngenbjets == 3,
            'ttcc': (ngenbjets == 2) & (ngencjets >= 2),
            'ttcj': (ngenbjets == 2) & (ngencjets == 1),
            'ttother': (ngenbjets == 2) & (ngencjets == 0) & (ngenlfjets >=2),
        }
            
        zeros = ak.zeros_like(events.nbJets, dtype=bool)
        isttbb_new, isttbj_new = zeros, zeros
        isttcc_new, isttcj_new = zeros, zeros
        isttother_new = zeros

        #category_genJets = ak.full(events.nbJets, 4)
        category_genJets = ak.Array(np.full(len(events.nbJets), 4))
        category_genJets = ak.to_numpy(
            ak.where(
                ttbar_criteria["ttbb"],
                0,
                category_genJets,
            )
        )
        category_genJets = ak.to_numpy(
            ak.where(
                ttbar_criteria["ttbj"],
                1,
                category_genJets,
            )
        )
        category_genJets = ak.to_numpy(
            ak.where(
                ttbar_criteria["ttcc"],
                2,
                category_genJets,
            )
        )
        category_genJets = ak.to_numpy(
            ak.where(
                ttbar_criteria["ttcj"],
                3,
                category_genJets,
            )
        )
            
            
        isttbb_new = ak.to_numpy(
            ak.where(
                ttbar_criteria['ttbb'],
                True,
                isttbb_new,
            )
        )
        isttbj_new = ak.to_numpy(
            ak.where(
                ttbar_criteria['ttbj'],
                True, 
                isttbj_new,
            )
        )
        isttcc_new = ak.to_numpy(
            ak.where(
                ttbar_criteria['ttcc'],
                True,
                isttcc_new,
            )
        )
        isttcj_new = ak.to_numpy(
            ak.where(
                ttbar_criteria['ttcj'],
                True,
                isttcj_new,
            )
        )
        isttother_new = ak.to_numpy(
            ak.where(
                (~isttbb_new & ~isttbj_new & ~isttcc_new & ~isttcj_new),
                #ttbar_criteria['ttother'],
                True,
                isttother_new,
            )
        )

        # genTtbarId category
        genTtbarId = events.genTtbarId

        ttbar_criteria2 = {
            'ttbb': (genTtbarId%100 == 53) | (genTtbarId%100 == 54) | (genTtbarId%100 == 55),
            'ttbj': (genTtbarId%100 == 51) | (genTtbarId%100 == 52),
            'ttcc': (genTtbarId%100 == 43) | (genTtbarId%100 == 44) | (genTtbarId%100 == 45),
            'ttcj': (genTtbarId%100 == 41) | (genTtbarId%100 == 42),
            'ttother': (genTtbarId%100 == 0),
        }
        #category_genTtbarId = ak.full(events.nbJets, 5)
        category_genTtbarId = ak.Array(np.full(len(events.nbJets), 5))
        category_genTtbarId = ak.to_numpy(
            ak.where(
                ttbar_criteria2["ttbb"],
                0,
                category_genTtbarId,
            )
        )
        category_genTtbarId = ak.to_numpy(
            ak.where(
                ttbar_criteria2["ttbj"],
                1,
                category_genTtbarId,
            )
        )
        category_genTtbarId = ak.to_numpy(
            ak.where(
                ttbar_criteria2["ttcc"],
                2,
                category_genTtbarId,
            )
        )
        category_genTtbarId = ak.to_numpy(
            ak.where(
                ttbar_criteria2["ttcj"],
                3,
                category_genTtbarId,
            )
        )
        category_genTtbarId = ak.to_numpy(
            ak.where(
                ttbar_criteria2["ttother"],
                4,
                category_genTtbarId,
            )
        )
            
        category_addJet = ak.Array(np.full(len(events.nbJets), 5))
        category_addJet = ak.to_numpy(
            ak.where(
                events.isttbb,
                0,
                category_addJet,
            )
        )
        category_addJet = ak.to_numpy(
            ak.where(
                events.isttbj,
                1,
                category_addJet,
            )
        )
        category_addJet = ak.to_numpy(
            ak.where(
                events.isttcc,
                2,
                category_addJet,
            )
        )
        category_addJet = ak.to_numpy(
            ak.where(
                events.isttcj,
                3,
                category_addJet,
            )
        )
        category_addJet = ak.to_numpy(
            ak.where(
                events.isttother,
                4,
                category_addJet,
            )
        )
            
            
        jet_drLep1 = ak.to_numpy(jets_bsort.delta_r(leptons[:,0]))
        jet_drLep2 = ak.to_numpy(jets_bsort.delta_r(leptons[:,1]))
        
        lep_drJet1 = np.column_stack((jet_drLep1[:, 0], jet_drLep2[:, 0]))
        lep_drJet2 = np.column_stack((jet_drLep1[:, 1], jet_drLep2[:, 1]))
        lep_drJet3 = np.column_stack((jet_drLep1[:, 2], jet_drLep2[:, 2]))
        lep_drJet4 = np.column_stack((jet_drLep1[:, 3], jet_drLep2[:, 3]))

        #lepton_drJet1 = transposed[:, 0]  # shape: (2,)
        #lepton_drJet2 = transposed[:, 1]  # shape: (2,)
        #lepton_drJet3 = transposed[:, 2]  # shape: (2,)
        #lepton_drJet4 = transposed[:, 3]  # shape: (2,)

        #lep_drJet1 = leptons.delta_r(jets_bsort[:,0])
        #lep_drJet2 = leptons.delta_r(jets_bsort[:,1])
        #lep_drJet3 = leptons.delta_r(jets_bsort[:,2])
        #lep_drJet4 = leptons.delta_r(jets_bsort[:,3])

        #pruned_ev = {'Channel': ak.to_numpy(events.Channel), 'nJets': ak.to_numpy(njets), 'nbJets': ak.to_numpy(events.nbJets), 'nbJets_T': ak.to_numpy(events.nbJets_T), 'ncJets': ak.to_numpy(events.ncJets), 'ncJets_T': ak.to_numpy(events.ncJets_T)}
        pruned_ev = {'sortJet': jets_bsort, 'Jet': jets[req_nbjet], 'Lepton': leptons, 'Channel': ak.to_numpy(events.Channel[req_nbjet]), 'nJets': ak.to_numpy(events.nJets[req_nbjet]), 'nbJets': ak.to_numpy(events.nbJets[req_nbjet]), 'nbJets_T': ak.to_numpy(events.nbJetsT[req_nbjet]), 'ncJets': ak.to_numpy(events.ncJets[req_nbjet]), 'ncJets_T': ak.to_numpy(events.ncJetsT[req_nbjet]), "ngenbJets": ak.to_numpy(ngenbjets[req_nbjet]), "ngencJets": ak.to_numpy(ngencjets[req_nbjet]), "ngenlfJets": ak.to_numpy(ngenlfjets[req_nbjet]), }

        # leptons
#        #pruned_ev['drLepton12'] = ak.to_numpy(dr_leptons)
        pruned_ev['massLepton12'] = ak.to_numpy(events.mll[req_nbjet])
        pruned_ev['Lepton_drJet1'] = ak.to_numpy(lep_drJet1)
        pruned_ev['Lepton_drJet2'] = ak.to_numpy(lep_drJet2)
        pruned_ev['Lepton_drJet3'] = ak.to_numpy(lep_drJet3)
        pruned_ev['Lepton_drJet4'] = ak.to_numpy(lep_drJet4)

        # MET
        pruned_ev['MET'] = ak.to_numpy(met.pt)
        pruned_ev['MET_phi'] = ak.to_numpy(met.phi)

        # jets
        pruned_ev['sortJet_drLep1'] = jet_drLep1
        pruned_ev['sortJet_drLep2'] = jet_drLep2

        # dijet
        pruned_ev['drJet12'] = ak.to_numpy(jets_bsort[:,0].delta_r(jets_bsort[:,1]))
        pruned_ev['drJet13'] = ak.to_numpy(jets_bsort[:,0].delta_r(jets_bsort[:,2]))
        pruned_ev['drJet14'] = ak.to_numpy(jets_bsort[:,0].delta_r(jets_bsort[:,3]))
        pruned_ev['drJet23'] = ak.to_numpy(jets_bsort[:,1].delta_r(jets_bsort[:,2]))
        pruned_ev['drJet24'] = ak.to_numpy(jets_bsort[:,1].delta_r(jets_bsort[:,3]))
        pruned_ev['drJet34'] = ak.to_numpy(jets_bsort[:,2].delta_r(jets_bsort[:,3]))

#        # btag sorting jets

        out_branch = ["events", "run", "luminosityBlock"]
        if isTTbar:
            pruned_ev.update({
                'isttbb': ak.to_numpy(events.isttbb[req_nbjet]), 
                'isttbj': ak.to_numpy(events.isttbj[req_nbjet]), 
                'isttcc': ak.to_numpy(events.isttcc[req_nbjet]), 
                'isttcj': ak.to_numpy(events.isttcj[req_nbjet]), 
                'isttother': ak.to_numpy(events.isttother[req_nbjet]),
                'isttbb_new': isttbb_new[req_nbjet], 
                'isttbj_new': isttbj_new[req_nbjet], 
                'isttcc_new': isttcc_new[req_nbjet], 
                'isttcj_new': isttcj_new[req_nbjet], 
                'isttother_new': isttother_new[req_nbjet],
                "category_genJets": category_genJets[req_nbjet],
                "category_genTtbarId": category_genTtbarId[req_nbjet],
                "genTtbarId": ak.to_numpy(genTtbarId[req_nbjet]),
		"category_addJet": category_addJet[req_nbjet],
		"nbJetsFromT": ak.to_numpy(events.nbJetsFromT[req_nbjet]),
		"nbJetsFromW": ak.to_numpy(events.nbJetsFromW[req_nbjet]),
		"ncJetsFromW": ak.to_numpy(events.ncJetsFromW[req_nbjet]),
		"naddbJets": ak.to_numpy(events.naddbJets[req_nbjet]),
		"naddcJets": ak.to_numpy(events.naddcJets[req_nbjet]),
                "bJetFromT": events.bJetFromT[req_nbjet],
                "bJetFromW": events.bJetFromW[req_nbjet],
                "cJetFromW": events.cJetFromW[req_nbjet],
                "addbJet": events.addbJet[req_nbjet],
                "addcJet": events.addcJet[req_nbjet],
                "addlfJet": events.addlfJets[req_nbjet],
                "nGenJets": ak.to_numpy(events.nGenJets[req_nbjet]),


            })

        if not isRealData:
            print("is not RealData")
            DeepJetC_totDown_weight, DeepJetC_totUp_weight = calc_tot_unc(events, "DeepJetCJet")
            DeepJetB_totDown_weight, DeepJetB_totUp_weight = calc_tot_unc(events, "DeepJetBJet")

            pruned_ev.update({
               "weight": ak.to_numpy(events.weight[req_nbjet]),
               "genweight_weight": ak.to_numpy(events.genweight.weight[req_nbjet]),
               "puweight_weight": ak.to_numpy(events.puweight.weight[req_nbjet]),
               "HLT_weight": ak.to_numpy(events.HLT.weight[req_nbjet]),
               "mu_Reco_weight": ak.to_numpy(events.mu.Reco_weight[req_nbjet]),
               "mu_ID_weight": ak.to_numpy(events.mu.ID_weight[req_nbjet]),
               "mu_Iso_weight": ak.to_numpy(events.mu.Iso_weight[req_nbjet]),
               "ele_Reco_weight": ak.to_numpy(events.ele.Reco_weight[req_nbjet]),
               "ele_ID_weight": ak.to_numpy(events.ele.ID_weight[req_nbjet]),
               "L1PreFiring_weight": ak.to_numpy(events.L1PreFiringWeight.Nom[req_nbjet]),
               "DeepJetC_weight": ak.to_numpy(ak.prod(events.DeepJetCJet.weight[req_nbjet], axis=-1)),
               "DeepJetB_weight": ak.to_numpy(ak.prod(events.DeepJetBJet.weight[req_nbjet], axis=-1)),

               # up
               "puweightUp_weight": ak.to_numpy(events.puweightUp.weight[req_nbjet]),
               "HLTUp_weight": ak.to_numpy(events.HLTUp.weight[req_nbjet]),
               "mu_RecoUp_weight": ak.to_numpy(events.mu.RecoUp_weight[req_nbjet]),
               "mu_IDUp_weight": ak.to_numpy(events.mu.IDUp_weight[req_nbjet]),
               "mu_IsoUp_weight": ak.to_numpy(events.mu.IsoUp_weight[req_nbjet]),
               "ele_RecoUp_weight": ak.to_numpy(events.ele.RecoUp_weight[req_nbjet]),
               "ele_IDUp_weight": ak.to_numpy(events.ele.IDUp_weight[req_nbjet]),
               "DeepJetC_TotUp_weight": DeepJetC_totUp_weight[req_nbjet],
               "DeepJetB_TotUp_weight": DeepJetB_totUp_weight[req_nbjet],
##               "DeepJetC_TotUp_weight": ak.to_numpy(DeepJetC_totUp_weight),
##               "DeepJetB_TotUp_weight": ak.to_numpy(DeepJetB_totUp_weight),
               # need to check
               "L1PreFiringUp_weight": ak.to_numpy(events.L1PreFiringWeight.Dn[req_nbjet]),
#
#
               # down
               "puweightDown_weight": ak.to_numpy(events.puweightDown.weight[req_nbjet]),
               "HLTDown_weight": ak.to_numpy(events.HLTDown.weight[req_nbjet]),
               "mu_RecoDown_weight": ak.to_numpy(events.mu.RecoDown_weight[req_nbjet]),
               "mu_IDDown_weight": ak.to_numpy(events.mu.IDDown_weight[req_nbjet]),
               "mu_IsoDown_weight": ak.to_numpy(events.mu.IsoDown_weight[req_nbjet]),
               "ele_RecoDown_weight": ak.to_numpy(events.ele.RecoDown_weight[req_nbjet]),
               "ele_IDDown_weight": ak.to_numpy(events.ele.IDDown_weight[req_nbjet]),
               "DeepJetC_TotDown_weight": DeepJetC_totDown_weight[req_nbjet],
               "DeepJetB_TotDown_weight": DeepJetB_totDown_weight[req_nbjet],
##               "DeepJetC_TotDown_weight": ak.to_numpy(DeepJetC_totDown_weight),
##               "DeepJetB_TotDown_weight": ak.to_numpy(DeepJetB_totDown_weight),
               # need to check
               "L1PreFiringDown_weight": ak.to_numpy(events.L1PreFiringWeight.Up[req_nbjet]),
#
            })

            for kin in ["pt", "eta"]:
                for obj in ["bJetFromT", "bJetFromW", "addbJet", "addcJet", "addlfJet"]:
                    out_branch = np.append(out_branch, [f"{obj}_{kin}"])
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

        with uproot.recreate( foutname) as fout:
            fout["Events"] = uproot_writeable(pruned_ev, include=out_branch)
        return {dataset: output}

    def postprocess(self, accumulator):
        return accumulator
