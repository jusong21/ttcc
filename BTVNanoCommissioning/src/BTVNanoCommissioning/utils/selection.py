import awkward as ak
import numpy as np

def jet_id(events, campaign):
    jetmask = (
        (events.Jet.pt > 25)
        & (abs(events.Jet.eta) < 2.4)
        & (events.Jet.jetId >= 2)
    )
    return jetmask

def ele_cuttightid(events, campaign):
    eleEtaGap  = (abs(events.Electron.eta) < 1.4442) | (abs(events.Electron.eta) > 1.566)
    elePassDXY = (abs(events.Electron.eta) <= 1.479) & (abs(events.Electron.dxy)<0.05) | (abs(events.Electron.eta)>1.479) & (abs(events.Electron.dxy) < 0.1)
    elePassDZ = (abs(events.Electron.eta) <= 1.479) & (abs(events.Electron.dz) < 0.1) | (abs(events.Electron.eta) > 1.479) & (abs(events.Electron.dz) < 0.2)
    elemask = (
        (events.Electron.pt > 20)
        & (abs(events.Electron.eta) < 2.4)
        & eleEtaGap
        & elePassDXY
        & elePassDZ
        & (events.Electron.cutBased >= 4)
    )
    return elemask

def mu_idiso(events, campaign):
    mumask = (
        (events.Muon.pt > 20)
        & (abs(events.Muon.eta) < 2.4)
        & events.Muon.tightId
        & (events.Muon.pfRelIso04_all < 0.15)
    )
    return mumask

btag_wp_dict = {
    "2016preVFP_UL": {
        "DeepFlav": {
            "b": {
                "L": 0.0505,
                "M": 0.2598,
                "T": 0.6502,
            },
            "c": {
                "L": [0.039, 0.327], # CvL, then CvB
                "M": [0.098, 0.370],
                "T": [0.270, 0.256],
            }
        }
    },
    "2016postVFP_UL": {
        "DeepFlav": {
            "b": {
                "L": 0.0480,
                "M": 0.2489,
                "T": 0.6377,
            },
            "c": {
                "L": [0.039, 0.305],
                "M": [0.099, 0.353],
                "T": [0.269, 0.247],
            }
        }
    },
    "2017_UL": {
        "DeepFlav": {
            "b": {
                "L": 0.532,
                "M": 0.3040,
                "T": 0.7476,
            },
            "c": {
                "L": [0.030, 0.400],
                "M": [0.085, 0.340],
                "T": [0.520, 0.050],
            }
        }
    },
    "2018_UL": {
        "DeepFlav": {
            "b": {
                "L": 0.0490,
                "M": 0.2783,
                "T": 0.7100,
            },
            "c": {
                "L": [0.038, 0.246],
                "M": [0.099, 0.325],
                "T": [0.282, 0.267],
            }
        }
    }
}

def btag_wp(jets, campaign, tagger, borc, wp):
    WP = btag_wp_dict[campaign]
    if borc == "b":
        jet_mask = jets[f"btag{tagger}B"] > WP[tagger]["b"][wp]
    else:
        jet_mask = (jets[f"btag{tagger}CvB"] > WP[tagger]["c"][wp][1]) & (
            jets[f"btag{tagger}CvL"] > WP[tagger]["c"][wp][0]
        )
    return jet_mask


def PNet_btag_category(jets, year):
    jet_b = jets.btagPNetB
    jet_cvl = jets.btagPNetCvL
    jet_c = jet_cvl * (1-jet_b)

    hvl = jet_b + jet_c
    bvc = jet_b / (jet_b + jet_c)
    
    criteria = {
        "2016": {
            "L0": hvl <= 0.1,
            "C0": (hvl > 0.1) & (hvl <= 0.17),
            "C1": (hvl > 0.17) & (hvl <= 0.35),
            "C2": (hvl > 0.35) & (bvc > 0.15) & (bvc <= 0.4),
            "C3": (hvl > 0.35) & (bvc > 0.05) & (bvc <= 0.15),
            "C4": (hvl > 0.35) & (bvc <= 0.05),
            "B0": (hvl > 0.35) & (bvc > 0.4) & (bvc <= 0.7),
            "B1": (hvl > 0.35) & (bvc > 0.7) & (bvc <= 0.88),
            "B2": (hvl > 0.35) & (bvc > 0.88) & (bvc <= 0.96),
            "B3": (hvl > 0.35) & (bvc > 0.96) & (bvc <= 0.99),
            "B4": (hvl > 0.35) & (bvc > 0.99),
        },
        "2017": {
            "L0": hvl <= 0.1,
            "C0": (hvl > 0.1) & (hvl <= 0.2),
            "C1": (hvl > 0.2) & (hvl <= 0.5),
            "C2": (hvl > 0.5) & (bvc > 0.15) & (bvc <= 0.4),
            "C3": (hvl > 0.5) & (bvc > 0.05) & (bvc <= 0.15),
            "C4": (hvl > 0.5) & (bvc <= 0.05),
            "B0": (hvl > 0.5) & (bvc > 0.4) & (bvc <= 0.7),
            "B1": (hvl > 0.5) & (bvc > 0.7) & (bvc <= 0.88),
            "B2": (hvl > 0.5) & (bvc > 0.88) & (bvc <= 0.96),
            "B3": (hvl > 0.5) & (bvc > 0.96) & (bvc <= 0.99),
            "B4": (hvl > 0.5) & (bvc > 0.99),
        },
        "2018": {
            "L0": hvl <= 0.1,
            "C0": (hvl > 0.1) & (hvl <= 0.2),
            "C1": (hvl > 0.2) & (hvl <= 0.5),
            "C2": (hvl > 0.5) & (bvc > 0.15) & (bvc <= 0.4),
            "C3": (hvl > 0.5) & (bvc > 0.05) & (bvc <= 0.15),
            "C4": (hvl > 0.5) & (bvc <= 0.05),
            "B0": (hvl > 0.5) & (bvc > 0.4) & (bvc <= 0.7),
            "B1": (hvl > 0.5) & (bvc > 0.7) & (bvc <= 0.88),
            "B2": (hvl > 0.5) & (bvc > 0.88) & (bvc <= 0.96),
            "B3": (hvl > 0.5) & (bvc > 0.96) & (bvc <= 0.99),
            "B4": (hvl > 0.5) & (bvc > 0.99),
        },
        "val": {
            "L0": 0,
            "C0": 40,
            "C1": 41,
            "C2": 42,
            "C3": 43,
            "C4": 44,
            "B0": 50,
            "B1": 51,
            "B2": 52,
            "B3": 53,
            "B4": 54,
        },
    }

    zeros = ak.zeros_like(jet_b, dtype=int)
    jet_category = zeros

    for wp in criteria[year].keys():

#        print('wp:', wp)
#        print(criteria[year][wp])
#        print(criteria["val"][wp])

        jet_category = ak.where(
            criteria[year][wp],
            criteria["val"][wp],
            jet_category,
        )

    return jet_category

def PNet_btag_wp(jet_category, flav, wp):
    wp_dict = {
        "b": {
            "L": (jet_category >= 50),
            "M": (jet_category >= 51),
            "T": (jet_category >= 52),
        },
        "c": {
            "L": (jet_category >= 40) & (jet_category < 45),
            "M": (jet_category >= 41) & (jet_category < 45),
            "T": (jet_category >= 42) & (jet_category < 45),
        },
    }
    wp_mask = wp_dict[flav][wp]
    return wp_mask

def ttbar_categorizer(genTtbarId):
    criteria = {
        "ttbb": {
            "cut": (genTtbarId%100 == 53) | (genTtbarId%100 == 54) | (genTtbarId%100 == 55),
            "cat": 0,
        },
        "ttbj": {
            "cut": (genTtbarId%100 == 51) | (genTtbarId%100 == 52),
            "cat": 1,
        },
        "ttcc": {
            "cut": (genTtbarId%100 == 43) | (genTtbarId%100 == 44) | (genTtbarId%100 == 45),
            "cat": 2,
        },
        "ttcj": {
            "cut": (genTtbarId%100 == 41) | (genTtbarId%100 == 42),
            "cat": 3,
        },
        "ttother": {
            "cut": (genTtbarId%100 == 0),
            "cat": 4,
        },
    }
    category = np.full(len(genTtbarId), -1, dtype=int)

    for tt in criteria.keys():

        category = ak.where(
            criteria[tt]["cut"],
            criteria[tt]["cat"],
            category,
        )

    return category



