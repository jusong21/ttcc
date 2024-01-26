from ROOT import *
from array import array

import uproot
import coffea.processor as processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import awkward as ak
import numpy as np
import hist
import matplotlib.pyplot as plt

finname = 'skimmed_nano.root'
fin = 'skimmed/'+finname
f = TFile(fin, 'READ')
tree = f.Get("Events")

btagPNetB = 0.2605
btagParTB = 0.451
btagPNetCvL = 0.160
btagPNetCvB = 0.304
btagParTCvL = 0.117
btagParTCvB = 0.128

def draw_plots( xval, yval, xcut, ycut, out ):

	xlim, ylim = 1, 1
	if 'pass' in xval: xlim= 9; ylim=9;

	gStyle.SetOptStat(0)
	# scat
	c1 = TCanvas('c1', '', 3)
	c1.SetLeftMargin(0.12)
	# logo & resp
	c2 = TCanvas('c2', '', 540, 500)
	c2.SetLeftMargin(0.12)
	c2.SetRightMargin(0.16)

	h_scat = TH2F('h_scat', '', 1000, 0, 1, 1000, 0, 1)
	h_lego = TH2F('h_lego', '', 100, 0, 1, 100, 0, 1)
	h_resp = TH2F('h_resp', '', 2, array('d', [0, xcut, xlim]), 2, array('d', [0, ycut, ylim]))
	h_list = [h_scat, h_lego, h_resp]

	cut = ''
	if 'bjet' in out: cut = 'Jet_hadronFlavour==5'
	elif 'cjet' in out: cut = 'Jet_hadronFlavour==4'
	elif 'lfjet' in out: cut = 'Jet_hadronFlavour==0'

	for h in h_list:
		name = h.GetName()
		h.SetXTitle(xval)
		h.SetYTitle(yval)
		h.SetMarkerSize(2.5)
		info = yval+':'+xval+'>>'+name

		if 'resp' in name:
			c2.cd()
			tree.Draw(info, cut, 'colz text')
			c2.Print(out+'_resp.pdf')
			c2.Clear()

		if 'pass' in xval: continue
		if 'scat' in name:
			c1.cd()
			tree.Draw(info, cut)

			lx = TLine(xcut, 0, xcut, 1)
			ly = TLine(0, ycut, 1, ycut)
			lx.SetLineColor(kRed)
			lx.SetLineWidth(3)
			ly.SetLineColor(kBlue)
			ly.SetLineWidth(3)
			lx.Draw()
			ly.Draw()

			c1.Print(out+'_scat.pdf')
			c1.Clear()

		if 'lego' in name:
			c2.cd()
			tree.Draw(info, cut, 'colz')
			c2.Print(out+'_lego.pdf')
			c2.Clear()


#draw_plots('Jet_btagPNetB', 'Jet_btagRobustParTAK4B', btagPNetB, btagParTB, 'h_alljet_btag')
#draw_plots('Jet_btagPNetB', 'Jet_btagRobustParTAK4B', btagPNetB, btagParTB, 'h_bjet_btag')
#draw_plots('Jet_btagPNetB', 'Jet_btagRobustParTAK4B', btagPNetB, btagParTB, 'h_cjet_btag')
draw_plots('Jet_btagPNetB', 'Jet_btagRobustParTAK4B', btagPNetB, btagParTB, 'h_lfjet_btag')


#draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', btagPNetCvL, btagPNetCvB, 'h_alljet_PNet_ctag')
#draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', btagPNetCvL, btagPNetCvB, 'h_bjet_PNet_ctag')
#draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', btagPNetCvL, btagPNetCvB, 'h_cjet_PNet_ctag')
#draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', btagPNetCvL, btagPNetCvB, 'h_lfjet_PNet_ctag')

#draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', btagParTCvL, btagParTCvB, 'h_alljet_ParT_ctag')
#draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', btagParTCvL, btagParTCvB, 'h_bjet_ParT_ctag')
#draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', btagParTCvL, btagParTCvB, 'h_cjet_ParT_ctag')
#draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', btagParTCvL, btagParTCvB, 'h_lfjet_ParT_ctag')

draw_plots('Jet_PNetCpass', 'Jet_ParTCpass', 3, 3, 'h_bjet_ctagged')
draw_plots('Jet_PNetCpass', 'Jet_ParTCpass', 3, 3, 'h_cjet_ctagged')
draw_plots('Jet_PNetCpass', 'Jet_ParTCpass', 3, 3, 'h_lfjet_ctagged')

###from datetime import datetime
###txt = open('results.txt', 'a')
###txt.write(datetime.now().isoformat(' ')+'\n')
###for h in h_btag+h_ctag_PNet+h_ctag_ParT:
###	if 'resp' in h.GetName():
###
###
###		noAll = h.GetBinContent(1,1)
###		noPNet_yesParT = h.GetBinContent(1,2)
###		yesPNet_noParT = h.GetBinContent(2,1)
###		yesAll = h.GetBinContent(2,2)
###		tot = h.GetEntries()
###		print(h.GetName())
###		
###		yesPNet = yesPNet_noParT + yesAll
###		yesParT = noPNet_yesParT + yesAll
###		
###		r_noAll = 			round(noAll*100 / tot, 1)
###		r_noPNet_yesParT =  round(noPNet_yesParT*100 / tot, 1)
###		r_yesPNet_noParT =  round(yesPNet_noParT*100 / tot, 1)
###		r_yesAll =          round(yesAll*100 / tot, 1)
###		r_yesPNet =         round(yesPNet*100 / tot, 1)
###		r_yesParT =         round(yesParT*100 / tot, 1)
###   	
###   	
###		txt = open('results.txt', 'a')
###		txt.write('** hist: '+h.GetName()+'\n')
###		txt.write('total: '+str(int(tot))+'\n')
###		txt.write('1 noAll: '+str(int(noAll))+' ('+str(r_noAll)+' %)'+'\n')
###		txt.write('2 rightDown: '+str(int(yesPNet_noParT))+' ('+str(r_yesPNet_noParT)+' %)'+'\n')
###		txt.write('3 leftUp: '+str(int(noPNet_yesParT))+' ('+str(r_noPNet_yesParT)+' %)'+'\n')
###		txt.write('4 yesAll: '+str(int(yesAll ))+' ('+str(r_yesAll)+' %)'+'\n')
###
###		txt.write('- yesPNet: '+str(int(yesPNet))+' ('+str(r_yesPNet)+' %) ovl '+str(round(yesAll*100/yesPNet, 1))+' %\n')
###		txt.write('- yesParT: '+str(int(yesParT))+' ('+str(r_yesParT)+' %) ovl '+str(round(yesAll*100/yesParT, 1))+' %\n')
###		txt.write('\n\n')
###
##
