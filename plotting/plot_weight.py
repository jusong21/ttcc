from locale import MON_1, MON_2
import ROOT
import numpy as np
import var_dict as dict
import json
import argparse

parser = argparse.ArgumentParser()
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
	help="channel",
	default=None,
	choices=[None, "ee", "mm", "em"]
)

args = parser.parse_args()
name = args.name
variables = args.variables
channel = args.channel

if channel==None:
	chan = "1"
	title = ""
else:
	if channel=="ee": chan = "Channel==-121"
	elif channel=="mm": chan = "Channel==-169"
	elif channel=="em": chan = "Channel==-143"
	title = channel

# Opening JSON file
jsonfile = open('samples.json')
 
# returns JSON object as 
# a dictionary
sampleinfo = json.load(jsonfile)

base_dir = "/pnfs/iihe/cms/store/user/jusong/ttcc/ntuples/v0/2017_UL/"
file = ROOT.TFile(base_dir+"TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/merged_nominal.root")
tree = file.Get("Events")


c = ROOT.TCanvas("c", "c", 3)
c.SetLogy()
ROOT.gStyle.SetOptStat(0)

#weight_list = ["puweight_weight", "HLT_weight", "mu_Reco_weight", "mu_ID_weight", "mu_Iso_weight", "ele_ID_weight", "ele_Reco_weight", "DeepJetC_weight"]
weight_list = ["puweight_weight", "DeepJetC_weight"]
#weight_list = ["puweight_weight"]

for weight in weight_list:
	c.SetLeftMargin( 0.12 )
	
	#if weight=="DeepJetC_weight": hist = ROOT.TH1F("h", "", 30, 0, 3)
	#elif weight=="puweight_weight": hist = ROOT.TH1F("h", "", 30, 0.6, 1.4)
	#else: hist = ROOT.TH1F("h", "", 30, 0.8, 1.2)
	draw_op = weight+">>h"
	tree.Draw(draw_op, chan)
	hist = ROOT.gROOT.FindObject("h")
	hist.SetXTitle(weight)
	hist.SetMaximum(hist.GetMaximum()*20)
	hist.SetYTitle("Events")
	hist.GetYaxis().SetTitleOffset(1.5)
	hist.SetTitle(title)
	hist.SetMarkerSize(.7)
	hist.Draw("text hist")

	c.SaveAs(weight+"_logy_"+name+".pdf")
	hist.Delete()
	#del hist

c.SetLeftMargin( 0.12 )
h_weight = ROOT.TH1F("h_w", "", 30, 0, 3)
h_weight_m = ROOT.TH1F("h_wm", "", 30, 0, 3)
h_weight_m.SetLineColor(ROOT.kBlack)

tree.Draw("weight/abs(genweight_weight)>>h_w", "genweight_weight>0&&"+chan)
tree.Draw("weight/genweight_weight>>h_wm", "genweight_weight<0&&"+chan)

h_weight.SetTitle(title)
h_weight.SetMaximum(h_weight.GetMaximum()*1.2)
h_weight.SetYTitle("Events")
h_weight.GetYaxis().SetTitleOffset(1.5)

tot_p = h_weight.GetEntries()
tot_m = h_weight_m.GetEntries()
print(tot_m)

h_weight_m.Scale(100)

h_weight.SetXTitle("Total weight")
h_weight.Draw()
h_weight_m.Draw("hist same")

line = ROOT.TLine(1, 0, 1, h_weight.GetMaximum())
line.SetLineColor(ROOT.kRed)
line.Draw("same")

t = ROOT.TLatex()
t.SetNDC()
t.SetTextSize(0.04); t.DrawLatex(0.6, 0.6, "N_{-}: "+str(int(tot_m)))
t.SetTextSize(0.04); t.DrawLatex(0.6, 0.65, "N_{+}: "+str(int(tot_p)))

leg = ROOT.TLegend( 0.4, 0.75, 0.8, 0.85)
leg.SetBorderSize(0)
leg.SetTextSize(0.035)
leg.AddEntry(h_weight, "(+) weighted events")
leg.AddEntry(h_weight_m, "(-) weighted events x 100")
leg.Draw("same")
c.Print("weight_"+name+".pdf")

#h_high = ROOT.TH1F("h_h", "", )
#h_low = ROOT.TH1F("h_l")
tree.Draw("weight/abs(genweight_weight)>>h_h", "weight/genweight_weight>2.5")
tree.Draw("weight/genweight_weight>>h_l", "weight/genweight_weight<0.3")

h_high = ROOT.gROOT.FindObject("h_h")
h_low = ROOT.gROOT.FindObject("h_l")

N_high = h_high.GetEntries()
N_low = h_low.GetEntries()

print("Weight > 2.5:", N_high)
print("Weight < 0.3:", N_low)

tree.Draw("weight/genweight_weight>>h_log")
h_log = ROOT.gROOT.FindObject("h_log")
h_log.SetMaximum(h_log.GetMaximum()*20)
#h_log.SetLogy()
c.SetLogy()
h_log.Draw()
c.Print("weight_logy_"+name+".pdf")


file.Close()

