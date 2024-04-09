import uproot
import coffea.processor as processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import mplhep as hep

finname = 'skimmed_nano.root'
fin = 'skimmed/'+finname
events = NanoEventsFactory.from_root(fin, schemaclass=NanoAODSchema).events()

plt.style.use(hep.cms.style.CMS)

r_label=None
l_label="Preliminary"
		
def get_ratio(jets, cvl, cvb, thresholds, cjet_condi, noncjet_condi):
	condis = []
	tprs = []
	fprs = []
	n_cjet = len(jets[cjet_condi])
	n_bjet = len(jets[noncjet_condi])
	for thre in thresholds:
		if len(cvl)==2: condi = (cvl[0] > thre) & (cvb[0] > thre) & (cvl[1] > thre) & (cvb[1] > thre)
		else: condi = (cvl > thre) & (cvb > thre)
		condis.append(condi)
	for condi in condis:
		n_cjet_ctagged = len(jets[condi&cjet_condi])
		n_bjet_ctagged = len(jets[condi&noncjet_condi])
		tpr = n_cjet_ctagged/n_cjet
		fpr = n_bjet_ctagged/n_bjet
		tprs.append(tpr)
		fprs.append(fpr)
	return tprs, fprs
		
def roc2d_plot(noncjet_flav, noncjet):
	jets = ak.flatten(events.Jet[(events.Jet.hadronFlavour==4) | (events.Jet.hadronFlavour==noncjet_flav)])

	cjet_condi = jets.hadronFlavour==4
	noncjet_condi = jets.hadronFlavour==noncjet_flav

	PNetCvB = jets.btagPNetCvB
	PNetCvL = jets.btagPNetCvL
	ParTCvB = jets.btagRobustParTAK4CvB
	ParTCvL = jets.btagRobustParTAK4CvL

	thresholds = np.linspace(0, 1, 200)[::-1]
	tprs_pnet, fprs_pnet = get_ratio(jets, PNetCvL, PNetCvB, thresholds, cjet_condi, noncjet_condi)
	tprs_part, fprs_part = get_ratio(jets, ParTCvL, ParTCvB, thresholds, cjet_condi, noncjet_condi)
	tprs_comb, fprs_comb = get_ratio(jets, [PNetCvL, ParTCvL], [PNetCvB, ParTCvB], thresholds, cjet_condi, noncjet_condi)

	plt.figure()
	plt.plot(tprs_pnet, fprs_pnet, label='PNet', color='C0')
	plt.plot(tprs_part, fprs_part, label='ParT', color='C1')
	plt.plot(tprs_comb, fprs_comb, label='PNet&ParT', color='C2')
	plt.xlabel('c-tagging efficiency')
	plt.ylabel(noncjet+' misId efficiency')
	plt.yscale('log')
	plt.xlim(0, 1)
	plt.ylim(2 * 1e-4, 1)
	plt.grid(which="minor", alpha=0.85)
	plt.grid(which="major", alpha=0.95, color="black")
	#plt.ylim([0.0001, 1])
	plt.legend(frameon=False)
	hep.cms.label(l_label, rlabel=r_label, com=13.6, year=2022)
	plt.savefig(noncjet+'_roc_curve2d.pdf', bbox_inches='tight')



def roc_c(jets, veto, disc):
	res = roc_curve(jets[veto].hadronFlavour, disc[veto], pos_label=4)
	return res

def roc1d_plot():
	jets = ak.flatten(events.Jet)
	veto_b = (jets.hadronFlavour==4) | (jets.hadronFlavour==0)
	veto_l = (jets.hadronFlavour==4) | (jets.hadronFlavour==5)
	PNetCvB = jets.btagPNetCvB
	ParTCvB = jets.btagRobustParTAK4CvB
	combCvB = 0.5 * (PNetCvB+ParTCvB)
	PNetCvL = jets.btagPNetCvL
	ParTCvL = jets.btagRobustParTAK4CvL
	combCvL = 0.5 * (PNetCvL+ParTCvL)
	roc_pnet = []
	roc_part = []
	roc_comb = []
	
	roc_pnet.append(roc_c(jets, veto_b, PNetCvL))
	roc_pnet.append(roc_c(jets, veto_l, PNetCvB))
	roc_part.append(roc_c(jets, veto_b, ParTCvL))
	roc_part.append(roc_c(jets, veto_l, ParTCvB))
	roc_comb.append(roc_c(jets, veto_b, combCvL))
	roc_comb.append(roc_c(jets, veto_l, combCvB))

#	for veto in [veto_b, veto_l]:
#		roc_pnet.append(roc_curve(jets[veto].hadronFlavour, PNetCvB[veto], pos_label=4))
#		roc_part.append(roc_curve(jets[veto].hadronFlavour, ParTCvB[veto], pos_label=4))
#		roc_comb.append(roc_curve(jets[veto].hadronFlavour, combCvB[veto], pos_label=4))
#	fig = plt.figure(figsize=(4.5,4.5))
	plt.figure()
	plt.plot(roc_pnet[0][1], roc_pnet[0][0], label='PNet CvL', color='C0', linestyle='dashed')
	plt.plot(roc_part[0][1], roc_part[0][0], label='ParT CvL', color='C1', linestyle='dashed')
	plt.plot(roc_comb[0][1], roc_comb[0][0], label='PNetxParT CvL', color='C2', linestyle='dashed')
	plt.plot(roc_pnet[1][1], roc_pnet[1][0], label='PNet CvB', color='C0')
	plt.plot(roc_part[1][1], roc_part[1][0], label='ParT CvB', color='C1')
	plt.plot(roc_comb[1][1], roc_comb[1][0], label='PNetxParT CvB', color='C2')
	plt.xlabel('c-tagging efficiency')
	plt.ylabel('misId efficiency')
#plt.ylim((0.0008, 1.1))
	plt.yscale('log')
	plt.xlim(0, 1)
	plt.ylim(2 * 1e-4, 1)
	plt.grid(which="minor", alpha=0.85)
	plt.grid(which="major", alpha=0.95, color="black")
	#plt.ylim([0.0001, 1])
	plt.legend(frameon=False)
	hep.cms.label(l_label, rlabel=r_label, com=13.6, year=2022)
	plt.savefig('roc_curve1d.pdf', bbox_inches='tight')


roc1d_plot()
#roc2d_plot(5, 'bjet')
#roc2d_plot(0, 'lfjet')




