import uproot
import coffea.processor as processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import awkward as ak
import numpy as np

# fname = 'nano_mcRun3_EE_1-1.root'
# finname = 'NANOv12_mcRun3_ee.root'
fname = 'nano.root'
fin = 'rootfiles/'+finname
events = NanoEventsFactory.from_root(fin, schemaclass=NanoAODSchema).events()


# object selection
# muons
muons = events.Muon
muon_cut = ( 
	(muons.pt > 20) 
	& (abs(muons.eta) < 2.4) 
	& (muons.pfRelIso04_all < 0.15) 
	& muons.tightId 
	)
muons = muons[muon_cut]

# electrons
electrons = events.Electron
eleEtaGap  = (abs(events.Electron.eta) < 1.4442) | (abs(events.Electron.eta) > 1.566)
elePassDXY = (abs(events.Electron.eta) < 1.479) & (abs(events.Electron.dxy)<0.05) | (abs(events.Electron.eta)>1.479) & (abs(events.Electron.dxy) < 0.1)
elePassDZ = (abs(events.Electron.eta) < 1.479) & (abs(events.Electron.dz) < 0.1) | (abs(events.Electron.eta) > 1.479) & (abs(events.Electron.dz) < 0.2)

ele_cut = (
	(events.Electron.pt > 20)
	& (abs(events.Electron.eta) < 2.1)
	& (events.Electron.cutBased >= 4)
	& eleEtaGap
	& elePassDXY
	& elePassDZ
	)
electrons = electrons[ele_cut]

jets = events.Jet[ 
	(events.Jet.pt > 30)
	& (abs(events.Jet.eta) < 2.4 )
	& ( events.Jet.jetId > 4 )
	& ak.all(events.Jet.metric_table(muons) > 0.4, axis=-1) 
	& ak.all(events.Jet.metric_table(electrons) > 0.4, axis=-1) ]

val_btagPNetCvL = 0.160
val_btagPNetCvB = 0.304
val_btagParTCvL = 0.117
val_btagParTCvB = 0.128

req_jets = ak.num(jets.pt)>=1
event_level = req_jets

sel_jets = jets[event_level]

jet_ctag_PNet_mask = (sel_jets.btagPNetCvL > val_btagPNetCvL) & (sel_jets.btagPNetCvB > val_btagPNetCvB) 
jet_ctag_ParT_mask = (sel_jets.btagRobustParTAK4CvL > val_btagParTCvL) & (sel_jets.btagRobustParTAK4CvB > val_btagParTCvB)

test_ev = {}
test_ev['Jet_pt'] = sel_jets.pt
test_ev['Jet_eta'] = sel_jets.eta
test_ev['Jet_jetId'] = sel_jets.jetId
test_ev['Jet_hadronFlavour'] = sel_jets.hadronFlavour
test_ev['Jet_btagPNetB'] = sel_jets.btagPNetB
test_ev['Jet_btagPNetCvB'] = sel_jets.btagPNetCvB
test_ev['Jet_btagPNetCvL'] = sel_jets.btagPNetCvL
test_ev['Jet_btagRobustParTAK4B'] = sel_jets.btagRobustParTAK4B
test_ev['Jet_btagRobustParTAK4CvB'] = sel_jets.btagRobustParTAK4CvB
test_ev['Jet_btagRobustParTAK4CvL'] = sel_jets.btagRobustParTAK4CvL
test_ev['Jet_PNetCpass'] = ak.mask(sel_jets, jet_ctag_PNet_mask).jetId
test_ev['Jet_ParTCpass'] = ak.mask(sel_jets, jet_ctag_ParT_mask).jetId

test_ev['Jet_PNetCpass'] = ak.fill_none(test_ev['Jet_PNetCpass'], 0)
test_ev['Jet_ParTCpass'] = ak.fill_none(test_ev['Jet_ParTCpass'], 0)


test_ev_record = ak.zip({key: val for key, val in test_ev.items()})

# strange, but it is needed to avoid an error message
print(test_ev.items())

foutname = 'skimmed_'+finname
with uproot.recreate('skimmed/'+foutname) as fout:
	fout['Events'] = test_ev_record
#
#val_btagPNetB = 0.2605
#val_btagParTB = 0.451
#val_btagPNetCvL = 0.160
#val_btagPNetCvB = 0.304
#val_btagParTCvL = 0.117
#val_btagParTCvB = 0.128
#
#btagPNetB = ak.flatten(jets.btagPNetB)
#btagParTB = ak.flatten(jets.btagRobustParTAK4B)
#btagPNetCvB = ak.flatten(jets.btagPNetCvB)
#btagParTCvB = ak.flatten(jets.btagRobustParTAK4CvB)
#btagPNetCvL = ak.flatten(jets.btagPNetCvL)
#btagParTCvL = ak.flatten(jets.btagRobustParTAK4CvL)
#
#h_btag = []
#h_ctag_PNet = []
#h_ctag_ParT = []
#h_ctagged = []
#
#jet_cat = ['alljet', 'bjet', 'cjet', 'lfjet']
#hist_style = ['scat', 'lego', 'resp']
#
#for jet in jet_cat:
#	for style in hist_style:
#		if style=='scat':
#			h_btag.append(TH2F('h_'+jet+'_btag_'+style, '', 1000, 0, 1, 1000, 0, 1))
#			h_ctag_PNet.append(TH2F('h_'+jet+'_PNet_ctag_'+style, '', 1000, 0, 1, 1000, 0, 1))
#			h_ctag_ParT.append(TH2F('h_'+jet+'_ParT_ctag_'+style, '', 1000, 0, 1, 1000, 0, 1))
#
#		if style=='lego':
#			h_btag.append(TH2F('h_'+jet+'_btag_'+style, '', 100, 0, 1, 100, 0, 1))
#			h_ctag_PNet.append(TH2F('h_'+jet+'_PNet_ctag_'+style, '', 100, 0, 1, 100, 0, 1))
#			h_ctag_ParT.append(TH2F('h_'+jet+'_ParT_ctag_'+style, '', 100, 0, 1, 100, 0, 1))
#
#		if style=='resp':
#			h_btag.append(TH2F('h_'+jet+'_btag_'+style, '', 2, array('d', [0, val_btagPNetB, 1]), 2, array('d', [0, val_btagParTB, 1])))
#			h_ctag_PNet.append(TH2F('h_'+jet+'_PNet_ctag_'+style, '', 2, array('d', [0, val_btagPNetCvL, 1]), 2, array('d', [0, val_btagPNetCvB, 1])))
#			h_ctag_ParT.append(TH2F('h_'+jet+'_ParT_ctag_'+style, '', 2, array('d', [0, val_btagParTCvL, 1]), 2, array('d', [0, val_btagParTCvB, 1])))
#
#h_bjet_ctagged = TH2F('h_bjet_ctagged_resp', '', 2, array('d', [0, 1, 3]), 2, array('d', [0, 1, 3]))
#h_cjet_ctagged = TH2F('h_cjet_ctagged_resp', '', 2, array('d', [0, 1, 3]), 2, array('d', [0, 1, 3]))
#
#jet_ctag_PNet_mask = (jets.btagPNetCvL > val_btagPNetCvL) & (jets.btagPNetCvB > val_btagPNetCvB) 
#jet_ctag_ParT_mask = (jets.btagRobustParTAK4CvL > val_btagParTCvL) & (jets.btagRobustParTAK4CvB > val_btagParTCvB)
#jet_flavour_mask = jets.hadronFlavour != 0
#jet_ctagged = jets[(jet_ctag_PNet_mask | jet_ctag_ParT_mask) & jet_flavour_mask]
#
#for jet in ak.flatten(jet_ctagged):
#	if (jet.btagPNetCvL > val_btagPNetCvL) & (jet.btagPNetCvB > val_btagPNetCvB) & (jet.btagRobustParTAK4CvL > val_btagParTCvL) & (jet.btagRobustParTAK4CvB > val_btagParTCvB):
##		print('all tagged')
##		print('btagPNetCvL ', jet.btagPNetCvL)
##		print('btagPNetCvB ', jet.btagPNetCvB)
##		print('btagParTCvL ', jet.btagRobustParTAK4CvL)
##		print('btagParTCvB ', jet.btagRobustParTAK4CvB)
##		print(jet.hadronFlavour,'\n')
#		if jet.hadronFlavour == 5:
#			h_bjet_ctagged.Fill(2,2)
#		elif jet.hadronFlavour == 4:
#			h_cjet_ctagged.Fill(2,2)
#	elif (jet.btagPNetCvL > val_btagPNetCvL) & (jet.btagPNetCvB > val_btagPNetCvB):
##		print('PNet')
##		print('btagPNetCvL ', jet.btagPNetCvL)
##		print('btagPNetCvB ', jet.btagPNetCvB)
##		print('btagParTCvL ', jet.btagRobustParTAK4CvL)
##		print('btagParTCvB ', jet.btagRobustParTAK4CvB)
##		print(jet.hadronFlavour,'\n')
#		if jet.hadronFlavour == 5:
#			h_bjet_ctagged.Fill(2,0)
#		elif jet.hadronFlavour == 4:
#			h_cjet_ctagged.Fill(2,0)
#	elif (jet.btagRobustParTAK4CvL > val_btagParTCvL) & (jet.btagRobustParTAK4CvB > val_btagParTCvB):
##		print('ParT')
##		print('btagPNetCvL ', jet.btagPNetCvL)
##		print('btagPNetCvB ', jet.btagPNetCvB)
##		print('btagParTCvL ', jet.btagRobustParTAK4CvL)
##		print('btagParTCvB ', jet.btagRobustParTAK4CvB)
##		print(jet.hadronFlavour,'\n')
#		if jet.hadronFlavour == 5:
#			h_bjet_ctagged.Fill(0,2)
#		elif jet.hadronFlavour == 4:
#			h_cjet_ctagged.Fill(0,2)
#
#
#
##for i in range(len(btagPNetB)):
### for i in range(1000):
##	for h in h_btag:
##		if 'alljet' in h.GetName():
##			h.Fill(btagPNetB[i], btagParTB[i])
##	for h in h_ctag_PNet:
##		if 'alljet' in h.GetName():
##			h.Fill(btagPNetCvL[i], btagPNetCvB[i])
##	for h in h_ctag_ParT:
##		if 'alljet' in h.GetName():
##			h.Fill(btagParTCvL[i], btagParTCvB[i])
##
##	if ak.flatten(jets.hadronFlavour)[i] == 5:
##		for h in h_btag:
##			if 'bjet' in h.GetName():
##				h.Fill(btagPNetB[i], btagParTB[i])
##		for h in h_ctag_PNet:
##			if 'bjet' in h.GetName():
##				h.Fill(btagPNetCvL[i], btagPNetCvB[i])
##		for h in h_ctag_ParT:
##			if 'bjet' in h.GetName():
##				h.Fill(btagParTCvL[i], btagParTCvB[i])
##
##	if ak.flatten(jets.hadronFlavour)[i] == 4:
##		for h in h_btag:
##			if 'cjet' in h.GetName():
##				h.Fill(btagPNetB[i], btagParTB[i])
##		for h in h_ctag_PNet:
##			if 'cjet' in h.GetName():
##				h.Fill(btagPNetCvL[i], btagPNetCvB[i])
##		for h in h_ctag_ParT:
##			if 'cjet' in h.GetName():
##				h.Fill(btagParTCvL[i], btagParTCvB[i])
##
##	if ak.flatten(jets.hadronFlavour)[i] == 0:
##		for h in h_btag:
##			if 'lfjet' in h.GetName():
##				h.Fill(btagPNetB[i], btagParTB[i])
##		for h in h_ctag_PNet:
##			if 'lfjet' in h.GetName():
##				h.Fill(btagPNetCvL[i], btagPNetCvB[i])
##		for h in h_ctag_ParT:
##			if 'lfjet' in h.GetName():
##				h.Fill(btagParTCvL[i], btagParTCvB[i])
#
##gStyle.SetOptStat(0)
##c1 = TCanvas('c1', '', 3)
##c1.SetLeftMargin(0.12)
#c2 = TCanvas('c2', '', 540, 500)
#c2.SetLeftMargin(0.12)
#c2.SetRightMargin(0.16)
#
#for h in [h_bjet_ctagged, h_cjet_ctagged]:
#
#	h.SetXTitle('PNet tagged')
#	h.SetYTitle('ParT tagged')
#	h.SetMarkerSize(2.5)
#	h.Draw('text colz')
#
#	c2.Print(h.GetName()+'.pdf')
#
##for h in h_btag:
##	h.SetXTitle('Jet_btagPNetB')
##	h.SetYTitle('Jet_btagRobustParTAK4B')
##	h.SetMarkerSize(2.5)
##	if 'scat' in h.GetName():
##		c1.cd()
##		h.Draw()
##		lx = TLine(val_btagPNetB, 0, val_btagPNetB, 1)	
##		ly = TLine(0, val_btagParTB, 1, val_btagParTB)	
##		lx.SetLineColor(kRed)	
##		lx.SetLineWidth(3)	
##		ly.SetLineColor(kBlue)	
##		ly.SetLineWidth(3)	
##		lx.Draw()	
##		ly.Draw()	
##		c1.Print(h.GetName()+'.pdf')
##	elif 'lego' in h.GetName():
##		c2.cd()
##		h.Draw('colz')
##		c2.Print(h.GetName()+'.pdf')
##	elif 'resp' in h.GetName():
##		c2.cd()
##		h.Draw('text colz')
##		c2.Print(h.GetName()+'.pdf')
##
##for h in h_ctag_PNet:
##	h.SetXTitle('Jet_btagPNetCvL')
##	h.SetYTitle('Jet_btagPNetCvB')
##	h.SetMarkerSize(2.5)
##	if 'scat' in h.GetName():
##		c1.cd()
##		h.Draw()
##		lx = TLine(val_btagPNetCvL, 0, val_btagPNetCvL, 1)	
##		ly = TLine(0, val_btagPNetCvB, 1, val_btagPNetCvB)	
##		lx.SetLineColor(kRed)	
##		lx.SetLineWidth(3)	
##		ly.SetLineColor(kBlue)	
##		ly.SetLineWidth(3)	
##		lx.Draw()	
##		ly.Draw()	
##		c1.Print(h.GetName()+'.pdf')
##	elif 'lego' in h.GetName():
##		c2.cd()
##		h.Draw('colz')
##		c2.Print(h.GetName()+'.pdf')
##	elif 'resp' in h.GetName():
##		c2.cd()
##		h.Draw('text colz')
##		c2.Print(h.GetName()+'.pdf')
##
##for h in h_ctag_ParT:
##	h.SetXTitle('Jet_btagRobustParTCvL')
##	h.SetYTitle('Jet_btagRobustParTCvB')
##	h.SetMarkerSize(2.5)
##	if 'scat' in h.GetName():
##		c1.cd()
##		h.Draw()
##		lx = TLine(val_btagParTCvL, 0, val_btagParTCvL, 1)	
##		ly = TLine(0, val_btagParTCvB, 1, val_btagParTCvB)	
##		lx.SetLineColor(kRed)	
##		lx.SetLineWidth(3)	
##		ly.SetLineColor(kBlue)	
##		ly.SetLineWidth(3)	
##		lx.Draw()	
##		ly.Draw()	
##		c1.Print(h.GetName()+'.pdf')
##	elif 'lego' in h.GetName():
##		c2.cd()
##		h.Draw('colz')
##		c2.Print(h.GetName()+'.pdf')
##	elif 'resp' in h.GetName():
##		c2.cd()
##		h.Draw('text colz')
##		c2.Print(h.GetName()+'.pdf')
##
##from datetime import datetime
##txt = open('results.txt', 'a')
##txt.write(datetime.now().isoformat(' ')+'\n')
##for h in h_btag+h_ctag_PNet+h_ctag_ParT:
##	if 'resp' in h.GetName():
##
##
##		noAll = h.GetBinContent(1,1)
##		noPNet_yesParT = h.GetBinContent(1,2)
##		yesPNet_noParT = h.GetBinContent(2,1)
##		yesAll = h.GetBinContent(2,2)
##		tot = h.GetEntries()
##		print(h.GetName())
##		
##		yesPNet = yesPNet_noParT + yesAll
##		yesParT = noPNet_yesParT + yesAll
##		
##		r_noAll = 			round(noAll*100 / tot, 1)
##		r_noPNet_yesParT =  round(noPNet_yesParT*100 / tot, 1)
##		r_yesPNet_noParT =  round(yesPNet_noParT*100 / tot, 1)
##		r_yesAll =          round(yesAll*100 / tot, 1)
##		r_yesPNet =         round(yesPNet*100 / tot, 1)
##		r_yesParT =         round(yesParT*100 / tot, 1)
##   	
##   	
##		txt = open('results.txt', 'a')
##		txt.write('** hist: '+h.GetName()+'\n')
##		txt.write('total: '+str(int(tot))+'\n')
##		txt.write('1 noAll: '+str(int(noAll))+' ('+str(r_noAll)+' %)'+'\n')
##		txt.write('2 rightDown: '+str(int(yesPNet_noParT))+' ('+str(r_yesPNet_noParT)+' %)'+'\n')
##		txt.write('3 leftUp: '+str(int(noPNet_yesParT))+' ('+str(r_noPNet_yesParT)+' %)'+'\n')
##		txt.write('4 yesAll: '+str(int(yesAll ))+' ('+str(r_yesAll)+' %)'+'\n')
##
##		txt.write('- yesPNet: '+str(int(yesPNet))+' ('+str(r_yesPNet)+' %) ovl '+str(round(yesAll*100/yesPNet, 1))+' %\n')
##		txt.write('- yesParT: '+str(int(yesParT))+' ('+str(r_yesParT)+' %) ovl '+str(round(yesAll*100/yesParT, 1))+' %\n')
##		txt.write('\n\n')
##
#
