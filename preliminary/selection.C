#include <string>
#include <iostream>
#include <fstream>

using namespace ROOT::VecOps;
using namespace std;
using rvec_f = RVec<float>;
using rvec_b = RVec<bool>;
using rvec_i = RVec<int>;

//void selection(int year, string path){
void selection(string outfile, int njet_cut, int nbjet_cut, int ncjet_cut){


	string sample;
	int year;
	float btag_m, cvsb_m, cvsl_m;
	
	ofstream out(outfile);
	if( out.is_open() ){
	
		//ifstream samples("samples.txt");
		ifstream samples("subdir/out.txt");
		while( getline(samples, sample) ){
			
			if( sample.find("UL18") ) year = 2018, btag_m = 0.277, cvsb_m = 0.29, cvsl_m = 0.085;
			else if( sample.find("UL17") ) year = 2017, btag_m = 0.3033, cvsb_m = 0.29, cvsl_m = 0.144;
			else if( sample.find("UL16") ) year = 2016, btag_m = 0.3093, cvsb_m = 0.29, cvsl_m = 0.085;
		
			 // HF tagging
			 auto btag = [btag_m](rvec_f jet_pt, rvec_f jet_eta, rvec_f jet_deepJet) { return jet_pt > 20 && abs(jet_eta) < 2.4 && jet_deepJet > btag_m; };
			 auto ctag = [cvsb_m, cvsl_m](rvec_f jet_pt, rvec_f jet_eta, rvec_f jet_cvsb, rvec_f jet_cvsl){ return jet_pt > 20 && abs(jet_eta) < 2.4 && jet_cvsb > cvsb_m && jet_cvsl > cvsl_m; };
			cout << year << endl;
			//inputfile = sample+"/NANOAODSM/106X_*/*/*.root";
			//inputfile = {"/pnfs/iihe/cms/ph/sc4/store/mc/RunIISummer20UL18NanoAODv9/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1_ext1-v2/80000/*root", "/pnfs/iihe/cms/ph/sc4/store/mc/RunIISummer20UL18NanoAODv9/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/*root"};
			//inputfile = {"/pnfs/iihe/cms/ph/sc4/store/mc/RunIISummer20UL18NanoAODv9/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1_ext1-v2/80000/*root"};
			//inputfile = {"/pnfs/iihe/cms/ph/sc4/store/mc/RunIISummer20UL18NanoAODv9/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1_ext1-v2/80000//CE3B46A4-C893-E647-9ADB-7285685CD832.root", "/pnfs/iihe/cms/ph/sc4/store/mc/RunIISummer20UL18NanoAODv9/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1_ext1-v2/80000//A58F5827-C5E8-6546-AEB4-8D942C8E4C8B.root"};
			//cout << inputfile << endl;
		
			cout << sample << endl;
			vector<string> inputfile = {sample};
			cout << sample << endl;
			ROOT::RDataFrame df("Events", inputfile);
		
			cout << "start" << endl;
			auto df_goodlepton = df.Filter("nMuon+nElectron==2", "nLep")
								  .Define("goodjets", "Jet_pt > 30 && abs(Jet_eta) < 2.4")
								  .Define("ngoodjets", "Sum(goodjets)")
								  .Define("bjets", btag, {"Jet_pt", "Jet_eta", "Jet_btagDeepFlavB"})
								  .Define("cjets", ctag, {"Jet_pt", "Jet_eta", "Jet_btagDeepFlavCvB", "Jet_btagDeepFlavCvL"})
								  .Define("nbjets", "Sum(bjets)")
							  	  .Define("ncjets", "Sum(cjets)");
		
			cout << "next" << endl;
			auto df_goodjet = df_goodlepton.Filter([njet_cut](int njets){return njets >= njet_cut;}, {"ngoodjets"}, "nJet");
			auto df_bjet = df_goodjet.Filter([nbjet_cut](int nbjets){return nbjets >= nbjet_cut;}, {"nbjets"},"nBJet");
			auto df_cjet = df_bjet.Filter([ncjet_cut](int ncjets){return ncjets >= ncjet_cut;}, {"ncjets"}, "nCJet");
		
			auto nevents = df_cjet.Count();
			auto report = df.Report();
		
			report->Print();
			auto pass = report->At("nJet").GetPass();
		
			cout << "pass: " << pass << endl;
			cout << "nevents: " << *nevents << endl;
		
			auto tot = report->At("nLep").GetAll();
			auto nlep_pass = report->At("nLep").GetPass();
			auto njet_pass = report->At("nJet").GetPass();
			auto nbjet_pass = report->At("nBJet").GetPass();
			auto ncjet_pass = report->At("nCJet").GetPass();
	
			out << sample << "," << tot << "," << nlep_pass << "," << njet_pass << "," << nbjet_pass << "," << ncjet_pass << endl;
		}
		out.close();
	}
}

