#include <iostream>
#include <TFile.h>
#include <TH2.h>

int testErr() {
    std::string f1_n = "TriggerSF_2017_ULv2.root";
    TFile *f1 = TFile::Open(f1_n.c_str());
    
    TH2F *h2_ee = (TH2F*)f1->Get("h2D_SF_ee_lepABpt_FullError");
    TH2F *h2_em = (TH2F*)f1->Get("h2D_SF_emu_lepABpt_FullError");
    TH2F *h2_mm = (TH2F*)f1->Get("h2D_SF_mumu_lepABpt_FullError");

    std::string f2_n = "TriggerSF_2017_ULv2_err.root";
    TFile *f2 = TFile::Open(f2_n.c_str());
    
    TH2F *h2_ee_err = (TH2F*)f2->Get("h2_ee_err");
    TH2F *h2_em_err = (TH2F*)f2->Get("h2_em_err");
    TH2F *h2_mm_err = (TH2F*)f2->Get("h2_mm_err");

    std::cout << "\nLooping over ee..." << std::endl;
    for (int xbin = 1; xbin <= h2_ee->GetNbinsX(); ++xbin) {
        for (int ybin = 1; ybin <= h2_ee->GetNbinsY(); ++ybin) {
            double err = h2_ee->GetBinError(xbin, ybin);
            double con = h2_ee_err->GetBinContent(xbin, ybin);
			std::cout << err << std::endl;
			std::cout << con << std::endl  << std::endl;
        }
    }

    std::cout << "\nLooping over em..." << std::endl;
    for (int xbin = 1; xbin <= h2_em->GetNbinsX(); ++xbin) {
        for (int ybin = 1; ybin <= h2_em->GetNbinsY(); ++ybin) {
            double err = h2_em->GetBinError(xbin, ybin);
            double con = h2_em_err->GetBinContent(xbin, ybin);
			std::cout << err << std::endl;
			std::cout << con << std::endl  << std::endl;
        }
    }

    std::cout << "\nLooping over mm..." << std::endl;
    for (int xbin = 1; xbin <= h2_mm->GetNbinsX(); ++xbin) {
        for (int ybin = 1; ybin <= h2_mm->GetNbinsY(); ++ybin) {
            double err = h2_mm->GetBinError(xbin, ybin);
            double con = h2_mm_err->GetBinContent(xbin, ybin);
			std::cout << err << std::endl;
			std::cout << con << std::endl  << std::endl;
        }
    }

    f1->Close();
    f2->Close();
    
    return 0;
}

