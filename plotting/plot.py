from locale import MON_1, MON_2
import ROOT
import numpy as np
import var_dict as dict
import json
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument(
	"-w",
	"--weight",
	dest="weight",
	#default="weight/abs(genweight_weight)",
	required=True,
	help="You can customaize weight in [weight, genweight_weight, puweight_weight, HLT_weight, mu_Reco_weight, mu_ID_weight, mu_Iso_weight, ele_ID_weight, ele_Reco_weight, DeepJetC_weight]"
)
parser.add_argument(
	"-n",
	"--name",
	dest="name",
	required=True,
	help="addition to output pdf"
)
parser.add_argument(
	"-v",
	"--vartiables",
	dest="variables",
	help="list of variables to be drawn"
)
parser.add_argument(
	"-c",
	"--channel",
	dest="channel",
	help="channel"
	default=None,
	choices=[None, "all", "ee", "mm", "em"]
)

args = parser.parse_args()
weight = args.weight
name = args.name
variables = args.variables
channel = args.channel

if channel==None:
	cut = weight
else:
	if channel=="ee": chan = "Channel==-121"
	elif channel=="mm": chan = "Channel==-169"
	elif channel=="em": chan = "Channel==-143"
	cut = weight+"&&"+chan

print("\nWeight will be applied:", weight)
print("Output pdf name: *_"+name+".pdf")
print("Channel:", channel)

# Opening JSON file
jsonfile = open('samples.json')
# returns JSON object as a dictionary
sampleinfo = json.load(jsonfile)

process_tags = ["WW_TuneCP5_13TeV-pythia8", "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8", "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8", "WZ_TuneCP5_13TeV-pythia8", "WZZ_TuneCP5_13TeV-amcatnlo-pythia8", "ZZ_TuneCP5_13TeV-pythia8", "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8", "TTToHadronic_TuneCP5_13TeV-powheg-pythia8", "ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8", "ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8", "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8", "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8", "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8", "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8", "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8", "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8", "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8", "TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8", "TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8", "TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8", "TTZToQQ_TuneCP5_13TeV-amcatnlo-pythia8", "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8", "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8", "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8", "ST_tW_antitop_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8", "ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8", "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8", "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8" ]

#base_dir = "/user/jusong/analysis/arrays_ttccNtuple_ttcc_ntuple_2017_local/"
base_dif = "/pnfs/iihe/cms/store/user/jusong/ttcc/ntuples/v0/"

data_file = ROOT.TFile(base_dir+"/Data/data_0.root")
data_tree = data_file.Get("Events")

trees = {}
files = []

print("\nLoading trees")
TT_trees = ROOT.TList()
for proc in process_tags:
	file = ROOT.TFile(base_dir+proc+"/merged_nominal.root")
	files.append(file)
	tree = file.Get("Events")

	if not tree:
		print(f"Error: 'Events' tree not found in file for process {proc}: {file_path}")
	elif not isinstance(tree, ROOT.TTree):
		print(f"Error: Object retrieved is not a TTree for process {proc}: {file_path}")
	else:
		trees[proc] = tree
		print(f"Successfully loaded 'Events' tree for process {proc}")

#vars = ["Jet1_pt", "nJets", "sortJet1_btagDeepFlavCvB", "sortJet1_btagDeepFlavCvL"]
if variables==None:
	vars = ["nJets"]
else: vars = variables
vars = dict.vardict.keys()
print(vars)

c = ROOT.TCanvas("c", "", 700, 1000)
c.SetFillColor(0)
c.SetBorderMode(0)
c.SetFrameFillStyle(0)
c.SetFrameBorderMode(0)
c.SetLeftMargin(0)
c.SetRightMargin(0)
c.SetTopMargin(0)
c.SetBottomMargin(0)

pad = ROOT.TPad("upper_pad", "", 0, 0.3, 1, 1.0)
pad.SetTickx(1)
pad.SetTicky(1)
pad.SetLeftMargin(0.15)
pad.SetRightMargin(0.1)
pad.SetTopMargin(0.1)
pad.SetBottomMargin(0.02)
pad.SetLogy()
pad.Draw()
c.cd()
pad2 = ROOT.TPad("lower_pad", "", 0, 0.0, 1, 0.3)
pad2.SetLeftMargin(0.15)
pad2.SetRightMargin(0.1)
pad2.SetTopMargin(0.02)
pad2.SetBottomMargin(0.36)
pad2.SetTickx(1)
pad2.SetTicky(1)
pad2.Draw()

for x in vars:

	print("\nStart to drawing", x)
	var = x
	outname = dict.vardict[x]["outname"]
	xtitle = dict.vardict[x]["xtitle"]
	nbins = dict.vardict[x]["nbins"]
	xlow = dict.vardict[x]["xlow"]
	xup = dict.vardict[x]["xup"]

	pad.cd()
	hists = {}
	for proc in process_tags:
		
		hists[proc] = ROOT.TH1F(proc, proc, nbins, xlow, xup)

	data_hist = ROOT.TH1F("Data", "Data", nbins, xlow, xup)
	Lumi = 41480.0
	
	glob_hist = ROOT.TH1F("glob_hist", "", nbins, xlow, xup)

	print("Filling histograms...")
	w=0
	i=0
	total_yield = 0
	for proc in process_tags:
		tree = trees.get(proc)
		hist = hists[proc]
		tree.Draw(var+">>"+hist.GetName(), cut )

#		print('\n'+proc.split('_')[0])
#		print("sum W:", hist.GetSumOfWeights(), "	inte:", hist.Integral())
#		print(hist.GetBinContent(1), " ", hist.GetBinError(1))
		hist.Scale(Lumi*sampleinfo[proc]["xsec"]/sampleinfo[proc]["events"])
#		print('After scaling')
#		print("sum W:", hist.GetSumOfWeights(), "	inte:", hist.Integral())

		print("Yield for " + proc + ": " + str(hist.Integral()))
		glob_hist.Add(hist)
		total_yield += hist.Integral()

		colour = sampleinfo[proc]["colour"]
		hist.SetLineColor(colour)
		hist.SetFillColorAlpha(colour, 1)

		w+=hist.GetSumOfWeights()
		i+=hist.Integral()

	ttHs = [hists[proc] for proc in process_tags if sampleinfo[proc]["legend"]=="ttH"]
	ttVs = [hists[proc] for proc in process_tags if sampleinfo[proc]["legend"]=="TTV"]
	DYs = [hists[proc] for proc in process_tags if sampleinfo[proc]["legend"]=="Z+jet"]
	VVs = [hists[proc] for proc in process_tags if sampleinfo[proc]["legend"]=="VV"]
	VVVs = [hists[proc] for proc in process_tags if sampleinfo[proc]["legend"]=="VVV"]
	STs = [hists[proc] for proc in process_tags if sampleinfo[proc]["legend"]=="ST"]
	WJs = [hists[proc] for proc in process_tags if sampleinfo[proc]["legend"]=="W+jet"]

	HIST_arr = [ttHs, ttVs, DYs, VVs, VVVs, STs, WJs]
	for HISTs in HIST_arr:
		for i, HIST in enumerate(HISTs):
			if i==0: proc = HIST.GetName(); continue
			hists[proc].Add(HIST)
			del hists[HIST.GetName()]

	print("Total yield: " + str(total_yield))

	stack = ROOT.THStack("hs", "")
	for key in hists:
		stack.Add(hists[key])

	stack.Draw("HIST")
	stack.GetXaxis().SetTitle("")
	stack.GetYaxis().SetTitle("Events")
	stack.GetYaxis().SetLabelSize(0.04)
	stack.GetYaxis().SetLabelOffset(0.009)
	stack.GetYaxis().SetTitleSize(0.05)
	stack.GetYaxis().SetTitleOffset(1.05)
	stack.GetXaxis().SetLabelSize(0)
	stack.GetXaxis().SetTitleSize(0.05)
	#stack.GetXaxis().SetLabelOffset(999)
	stack.GetYaxis().SetMaxDigits(3)
	stack.GetXaxis().SetNdivisions(505)
	#stack.GetYaxis().SetMaximum(stack.GetMaximum()*8)
	stack.SetMaximum(stack.GetMaximum()*20)
	#stack.SetMaximum(stack.GetMaximum()*2)
	#stack.SetMinimum(0)
	stack.SetMinimum(1)

	#glob_hist = stack.GetStack().Last()
	#glob_hist.SetLineWidth(0)
	#glob_hist.SetFillColor(12)
	#glob_hist.SetFillStyle(3104)
	#glob_hist.Draw("SAME E2")

	#ROOT.gStyle.SetEndErrorSize(4)
	err_band = ROOT.TGraphErrors(glob_hist)
	err_band.SetFillColor(ROOT.kBlue+3)
	err_band.SetFillStyle(3154)
	err_band.SetMarkerSize(0)
	err_band.Draw("SAME e2")

	#ROOT.gStyle.SetErrorX(0)
	ROOT.gStyle.SetErrorX(0)
	data_tree.Draw(var+">>"+data_hist.GetName(), "", "SAME P E")
	#data_tree.Draw(var+">>"+data_hist.GetName())
	data_hist.SetLineColor(sampleinfo["Data"]["colour"])
	data_hist.SetMarkerStyle(20)

	legend = ROOT.TLegend(0.3, 0.7, 0.83, 0.87)
	legend.SetNColumns(3)
	for key in hists:
		#print('key ', key, ' legend ', sampleinfo[key]["legend"])
		legend.AddEntry(hists[key], sampleinfo[key]["legend"])
	legend.AddEntry(data_hist, sampleinfo["Data"]["legend"])
	legend.AddEntry(err_band, "Uncertainty")

	legend.SetTextFont(42)
	legend.SetFillStyle(0)
	legend.SetBorderSize(0)
	legend.SetTextSize(0.045)


	label = ROOT.TLatex()
	label.SetNDC()
	label.SetTextSize(0.055); label.SetTextFont(61); label.DrawLatex(0.15,0.915, "CMS")
	label.SetTextSize(0.035); label.SetTextFont(50); label.DrawLatex(0.245,0.915, "Simulation")
	label.SetTextSize(0.045); label.SetTextFont(42); label.DrawLatex(0.62, 0.915, str(41.48)+" fb^{-1} (13 TeV)")

	pad.RedrawAxis()
	legend.Draw("SAME")

	pad2.cd()

	ratio = data_hist.Clone()
	ratio.Divide(glob_hist)
	for i in range(1, ratio.GetNbinsX()+1):
		try:
			r_cont = data_hist.GetBinError(i)/data_hist.GetBinContent(i)
		except:
			r_cont = 0
		ratio.SetBinError(i, r_cont)
		#ratio.SetBinError(i, data_hist.GetBinError(i)/data_hist.GetBinContent(i))

#	ratio.Reset()
#	for i in range(1, ratio.GetNbinsX()+1):
#		data_cont = data_hist.GetBinContent(i)
#		mc_cont = stack.GetStack().Last().GetBinContent(i)
#		try:
#			r_cont = float(data_cont) / float(mc_cont)
#		except:
#			r_cont = 0
#		ratio.SetBinContent(i, r_cont)

	ratio.GetYaxis().CenterTitle()
	ratio.GetYaxis().SetTitleSize(0.11)
	ratio.GetYaxis().SetLabelOffset(0.004)
	ratio.GetYaxis().SetTitleOffset(0.5)
	ratio.GetYaxis().SetLabelSize(.09)
	ratio.GetYaxis().SetNdivisions(505)
	
	#ratio.GetXaxis().SetTickSize(0.5)
	ratio.GetXaxis().SetTitleSize(0.11)
	ratio.GetXaxis().SetLabelSize(0.1)
	ratio.GetXaxis().SetTitleOffset(1.)
	ratio.GetXaxis().SetLabelOffset(0.006)
	ratio.GetXaxis().SetNdivisions(505)
	ratio.GetXaxis().SetTickLength(0.09)
	ratio.GetYaxis().SetTickLength(0.03)
	
	#ratio.SetYTitle("#frac{Data}{MC}")
	ratio.SetYTitle("Data / MC")
	ratio.SetXTitle(xtitle)
	ratio.SetTitle("")
	ratio.SetStats(0)
	ratio.SetMarkerStyle(20)
	ratio.SetMinimum(0.5)
	ratio.SetMaximum(1.5)
	ratio.Draw("P E")

#	h_err = ROOT.TH1F("h_err", "", nbins, xlow, xup)
	#h = ROOT.TH1F("h", "", nbins, 
	h_err = ROOT.TGraphErrors(nbins)
	x_err = (abs(ratio.GetBinCenter(2))-abs(ratio.GetBinCenter(1)))/2
	print(x_err)
	for i in range(1, ratio.GetNbinsX()+1):
		center = ratio.GetBinCenter(i)
		h_err.SetPoint(i, center, 1)
		#h_err.AddPointError(center, 1, 0.5, glob_hist.GetBinError(i)/glob_hist.GetBinContent(i))
		try:
			h_cont = glob_hist.GetBinError(i)/glob_hist.GetBinContent(i)
		except:
			h_cont = 0
		#h_err.SetPointError(i, 0.5, glob_hist.GetBinError(i)/glob_hist.GetBinContent(i))
		h_err.SetPointError(i, x_err, h_cont)
		print(i, center, h_cont)
#		h_err.SetBinContent(i, 1)
#	for i in range(1, glob_hist.GetNbinsX()+1):
#		print(i, glob_hist.GetBinError(i)/glob_hist.GetBinContent(i))
#		h_err.SetBinContent(i, 1)
#		h_err.SetBinError(i, glob_hist.GetBinError(i)/glob_hist.GetBinContent(i))

	#h_err.SetLineWidth(2)
	h_err.SetFillColor(ROOT.kBlue+3)
	h_err.SetFillStyle(3154)
	h_err.Draw("SAME E2")


	line = ROOT.TLine(xlow, 1, xup, 1)
	line.SetLineColor(ROOT.kRed)
	#line.Draw("SAME")

	pad2.SetGridy()
	pad2.RedrawAxis()

	c.cd()
	c.Modified()
	c.Update()
	c.RedrawAxis()
	ROOT.gPad.Update()
	ROOT.gPad.RedrawAxis()
	
	c.SaveAs("plots/0618/"+outname+'_'+name+'.pdf')

	pad.Clear()
	pad2.Clear()
	for proc in process_tags:
		ROOT.gDirectory.Delete(proc)
	del hists, stack, data_hist, ratio

for file in files:
	file.Close()
data_file.Close()

