#include <iostream>
#include <TFile.h>
#include <TH2.h>

int saveErrPlot() {
    std::string f_name = "TriggerSF_2017_ULv2.root";
    TFile *file = TFile::Open(f_name.c_str());
    
    TH2F *h2_ee = (TH2F*)file->Get("h2D_SF_ee_lepABpt_FullError");
    TH2F *h2_em = (TH2F*)file->Get("h2D_SF_emu_lepABpt_FullError");
    TH2F *h2_mm = (TH2F*)file->Get("h2D_SF_mumu_lepABpt_FullError");

    TH2F *h2_ee_err = (TH2F*)h2_ee->Clone("h2_ee_err");
    TH2F *h2_em_err = (TH2F*)h2_em->Clone("h2_em_err");
    TH2F *h2_mm_err = (TH2F*)h2_mm->Clone("h2_mm_err");

    std::cout << "\nLooping over ee..." << std::endl;
    for (int xbin = 1; xbin <= h2_ee->GetNbinsX(); ++xbin) {
        for (int ybin = 1; ybin <= h2_ee->GetNbinsY(); ++ybin) {
            double err = h2_ee->GetBinError(xbin, ybin);
            h2_ee_err->SetBinContent(xbin, ybin, err);
            std::cout << xbin << " " << ybin << " " << err << " " << h2_ee_err->GetBinContent(xbin, ybin) << std::endl;
        }
    }
    
    std::cout << "\nLooping over em..." << std::endl;
    for (int xbin = 1; xbin <= h2_em->GetNbinsX(); ++xbin) {
        for (int ybin = 1; ybin <= h2_em->GetNbinsY(); ++ybin) {
            double err = h2_em->GetBinError(xbin, ybin);
            h2_em_err->SetBinContent(xbin, ybin, err);
            std::cout << xbin << " " << ybin << " " << err << " " << h2_em_err->GetBinContent(xbin, ybin) << std::endl;
        }
    }
    
    std::cout << "\nLooping over mm..." << std::endl;
    for (int xbin = 1; xbin <= h2_mm->GetNbinsX(); ++xbin) {
        for (int ybin = 1; ybin <= h2_mm->GetNbinsY(); ++ybin) {
            double err = h2_mm->GetBinError(xbin, ybin);
            h2_mm_err->SetBinContent(xbin, ybin, err);
            std::cout << xbin << " " << ybin << " " << err << " " << h2_mm_err->GetBinContent(xbin, ybin) << std::endl;
        }
    }
    
    std::cout << "Writing histograms..." << std::endl;
    
    TFile *f_out = TFile::Open("TriggerSF_2017_ULv2_err.root", "RECREATE");
    h2_ee_err->Write();
    h2_em_err->Write();
    h2_mm_err->Write();
    f_out->Close();
    
    file->Close();
    
    return 0;
}

