import uproot
import coffea.processor as processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import awkward as ak
import numpy as np

fin = 'arrays_ttcc2L2Nu_ttcc_2L2Nu_2017_1/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/f1023_nominal_0.root'
events = NanoEventsFactory.from_root(fin, schemaclass=NanoAODSchema).events()
events = events[:10]


uncs = {
	"DeepJetCJet": {
		"Up": {
			"br": "DeepJetC_Up_tot",
		 	"list": ["ExtrapUp_weight", "InterpUp_weight", "LHEScaleWeight_muFUp_weight", "LHEScaleWeight_muRUp_weight", "PSWeightFSRUp_weight", "PSWeightISRUp_weight", "PUWeightUp_weight", "StatUp_weight", "XSec_BRUnc_DYJets_bUp_weight", "XSec_BRUnc_DYJets_cUp_weight", "XSec_BRUnc_WJets_cUp_weight", "jerUp_weight", "jesTotalUp_weight"],
		},
		"Down": {
			"br": "Down_tot",
		 	"list": ["ExtrapDown_weight", "InterpDown_weight", "LHEScaleWeight_muFDown_weight", "LHEScaleWeight_muRDown_weight", "PSWeightFSRDown_weight", "PSWeightISRDown_weight", "PUWeightDown_weight", "StatDown_weight", "XSec_BRUnc_DYJets_bDown_weight", "XSec_BRUnc_DYJets_cDown_weight", "XSec_BRUnc_WJets_cDown_weight", "jerDown_weight", "jesTotalDown_weight"],
		}
	}
}

print('nom deepJetC weight')
print(events['DeepJetCJet']['weight'])
nom=events['DeepJetCJet']['weight']

DeepJet = "DeepJetCJet"
for unc in uncs[DeepJet]["Up"]["list"]:
	#up = ak.to_numpy(events[DeepJet][unc])
	print(unc)
	print(events[DeepJet][unc]-nom)
	print()

for unc in uncs[DeepJet]["Down"]["list"]:
	#up = ak.to_numpy(events[DeepJet][unc])
	print(unc)
	print(nom-events[DeepJet][unc])
	print()
#	up = ak.to_numpy(ak.flatten(events[DeepJet][unc]))
#	counts = ak.num(events[DeepJet][unc])
#	sig = up-nom
#	sq_sig = np.square(sig)
#	sq_sig_ak = ak.unflatten(sq_sig, counts)
#	sum_sq_sig = ak.to_numpy(ak.sum(sq_sig_ak, axis=-1))
#	sq_up_tot += sum_sq_sig
	
