from ROOT import *
from array import array

#f = TFile("nano_mcRun3_EE_1-1.root", 'READ')
f = TFile("NANOv12_mcRun3_ee.root", 'READ')
tree = f.Get("Events")

def draw_plots( xval, yval, xcut, ycut, cut, out ):

	# scatter plot
	gStyle.SetOptStat(0)
	c1 = TCanvas('c', 'c', 3)
	c1.SetLeftMargin(0.15)
	h1 = TH2F('h1', '', 1000, 0, 1, 1000, 0, 1)
	h1.SetXTitle(xval)
	h1.SetYTitle(yval)

	info = yval+':'+xval+'>>h1'
	if cut=='':
		cut = 'Jet_pt>20&abs(Jet_eta)<2.5&Jet_btagPNetB>0&Jet_btagRobustParTAK4B>0'
	else: cut = cut+'&Jet_pt>20&abs(Jet_eta)<2.5&Jet_btagPNetB>0&Jet_btagRobustParTAK4B>0'
	tree.Draw(info, cut)

	if not xcut=='':
		print('in if')
		lx = TLine(xcut, 0, xcut, 1)
		ly = TLine(0, ycut, 1, ycut)
	
		lx.SetLineColor(kRed)
		lx.SetLineWidth(3)
		ly.SetLineColor(kBlue)
		ly.SetLineWidth(3)
		lx.Draw()
		ly.Draw()
	
	c1.Print('scatter_'+out+'.pdf')

	# response matrix
	c2 = TCanvas('c', 'c', 440, 410)
	c2.SetLeftMargin(0.15)
	c2.SetRightMargin(0.16)
	if not xcut=='':
		h2 = TH2F('h2', '', 2, array('d', [0, xcut, 1]), 2, array('d', [0, ycut, 1]))
		h2.SetXTitle(xval)
		h2.SetYTitle(yval)
		h2.SetMarkerSize(2.5)
		info = yval+':'+xval+'>>h2'
		tree.Draw(info, cut, 'colz text')

		tot = h2.GetEntries()
		bincon22 = h2.GetBinContent(2,2) + h2.GetBinContent(3,3)+h2.GetBinContent(2,3)+h2.GetBinContent(3,2)

		h2.SetBinContent(2, 2, bincon22)
		h2.Draw('colz text')
		
		c2.Print('response_'+out+'.pdf')

		noAll = h2.GetBinContent(1,1)
		noPNet_yesParT = h2.GetBinContent(1,2)
		yesPNet_noParT = h2.GetBinContent(2,1)
		yesAll = h2.GetBinContent(2,2)
		#tot = noAll + noPNet_yesParT + yesPNet_noParT + yesAll
		
		yesPNet = yesPNet_noParT + yesAll
		yesParT = noPNet_yesParT + yesAll
		
		r_noAll = 			round(noAll*100 / tot, 1)
		r_noPNet_yesParT =  round(noPNet_yesParT*100 / tot, 1)
		r_yesPNet_noParT =  round(yesPNet_noParT*100 / tot, 1)
		r_yesAll =          round(yesAll*100 / tot, 1)
		r_yesPNet =         round(yesPNet*100 / tot, 1)
		r_yesParT =         round(yesParT*100 / tot, 1)
   	
   	
		txt = open('results.txt', 'a')
		txt.write('** mode: '+out+'\n')
		txt.write('total: '+str(int(tot))+'\n')
		txt.write('1 noAll: '+str(int(noAll))+' ('+str(r_noAll)+' %)'+'\n')
		txt.write('2 yesPNet_noParT: '+str(int(yesPNet_noParT))+' ('+str(r_yesPNet_noParT)+' %)'+'\n')
		txt.write('3 noPNet_yesParT: '+str(int(noPNet_yesParT))+' ('+str(r_noPNet_yesParT)+' %)'+'\n')
		txt.write('4 yesAll: '+str(int(yesAll ))+' ('+str(r_yesAll)+' %)'+'\n')

		txt.write('- yesPNet: '+str(int(yesPNet))+' ('+str(r_yesPNet)+' %) ovl '+str(round(yesAll*100/yesPNet, 1))+' %\n')
		txt.write('- yesParT: '+str(int(yesParT))+' ('+str(r_yesParT)+' %) ovl '+str(round(yesAll*100/yesParT, 1))+' %\n')
		txt.write('\n\n')

	# lego plot
	c2.Clear()
	h3 = TH2F('h3', '', 100, 0, 1, 100, 0, 1)
	h3.SetXTitle(xval)
	h3.SetYTitle(yval)
	info = yval+':'+xval+'>>h3'
	tree.Draw(info, cut, 'colz')
	c2.Print('lego_'+out+'.pdf')


# btagging
# all jets
draw_plots('Jet_btagPNetB', 'Jet_btagRobustParTAK4B', 0.2605, 0.451, '', 'btag')

# b jets
draw_plots('Jet_btagPNetB', 'Jet_btagRobustParTAK4B', 0.2605, 0.451, 'Jet_hadronFlavour==5', 'btag_bjet')
#
## c jets
draw_plots('Jet_btagPNetB', 'Jet_btagRobustParTAK4B', 0.2605, 0.451, 'Jet_hadronFlavour==4', 'btag_cjet')
#
## lf jets
draw_plots('Jet_btagPNetB', 'Jet_btagRobustParTAK4B', 0.2605, 0.451, 'Jet_hadronFlavour==0', 'btag_lfjet')
#
#
## CvL CvB
draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', 0.160, 0.304, '', 'PNetC')
draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', 0.117, 0.128, '', 'ParTC')
#
draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', 0.160, 0.304, 'Jet_hadronFlavour==5', 'PNetC_bjet')
draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', 0.117, 0.128, 'Jet_hadronFlavour==5', 'ParTC_bjet')
#
draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', 0.160, 0.304, 'Jet_hadronFlavour==4', 'PNetC_cjet')
draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', 0.117, 0.128, 'Jet_hadronFlavour==4', 'ParTC_cjet')
#
draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', 0.160, 0.304, 'Jet_hadronFlavour==0', 'PNetC_lfjet')
draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', 0.117, 0.128, 'Jet_hadronFlavour==0', 'ParTC_lfjet')
#
## CvL CvB other tagger tagged
draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', 0.160, 0.304, 'Jet_btagRobustParTAK4CvL>0.117 & Jet_btagRobustParTAK4CvB>0.128', 'PNetC_ParTctagged')
draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', 0.117, 0.128, 'Jet_btagPNetCvL>0.160 & Jet_btagPNetCvB>0.304', 'ParTC_PNetctagged')
#
draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', 0.160, 0.304, 'Jet_hadronFlavour==5 & Jet_btagRobustParTAK4CvL>0.117 & Jet_btagRobustParTAK4CvB>0.128', 'PNetC_ParTctagged_bjet')
draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', 0.117, 0.128, 'Jet_hadronFlavour==5 & Jet_btagPNetCvL>0.160 & Jet_btagPNetCvB>0.304', 'ParTC_PNetctagged_bjet')
#
draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', 0.160, 0.304, 'Jet_hadronFlavour==4 & Jet_btagRobustParTAK4CvL>0.117 & Jet_btagRobustParTAK4CvB>0.128', 'PNetC_ParTctagged_cjet')
draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', 0.117, 0.128, 'Jet_hadronFlavour==4 & Jet_btagPNetCvL>0.160 & Jet_btagPNetCvB>0.304', 'ParTC_PNetctagged_cjet')
#
draw_plots('Jet_btagPNetCvL', 'Jet_btagPNetCvB', 0.160, 0.304, 'Jet_hadronFlavour==0 & Jet_btagRobustParTAK4CvL>0.117 & Jet_btagRobustParTAK4CvB>0.128', 'PNetC_ParTctagged_lfjet')
draw_plots('Jet_btagRobustParTAK4CvL', 'Jet_btagRobustParTAK4CvB', 0.117, 0.128, 'Jet_hadronFlavour==0 & Jet_btagPNetCvL>0.160 & Jet_btagPNetCvB>0.304', 'ParTC_PNetctagged_lfjet')
#
