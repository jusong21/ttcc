import numpy as np
from rich.progress import track

from typing import Dict, List


def weight_all_files_histrogram_weighting(
    files,
    histograms: Dict = None,
    reference_key="isB",
    #bins_pt=None,
    #bins_eta=None,
    bins_nbjets=None,
    bins_ncjets=None,
) -> List[str]:
    # for file in track(files, "Evaluating and saving the weights..."):
    for file in files:
        samples = np.load(file, allow_pickle=True)
#        jet_pt = samples["global_features"]["jet_pt"]
#        jet_eta = samples["global_features"]["jet_eta"]
#        pt_coordinate = np.digitize(jet_pt, bins_pt) - 1
#        eta_coordinate = np.digitize(jet_eta, bins_eta) - 1
#        weights = histogram_weighting(histograms, reference_key=reference_key)

        nbjets = samples["global_features"]["nbJets"]
        ncjets = samples["global_features"]["ncJets"]
        nbjets_coordinate = np.digitize(nbjets, bins_nbjets) - 1
        ncjets_coordinate = np.digitize(ncjets, bins_ncjets) - 1
        weights = histogram_weighting(histograms, reference_key=reference_key)

        # values need to be casted to list first!
        # otherwise it will be an object array
        flavour_idx = np.lib.recfunctions.apply_along_fields(
            np.argmax, samples["truth"]
        )
        weights = np.array(list(weights.values()))
        w = weights[flavour_idx, nbjets_coordinate, ncjets_coordinate]

        np.savez(file, **samples, weight=w)
    return files


def histogram_weighting(
    histograms: Dict,
    reference_key="isttcc",
) -> Dict:
    # Get the weights
    reference_histogram = histograms[reference_key].view() / np.max(
        histograms[reference_key].view()
    )
    weights_list = {}
    for key, hist in histograms.items():
        other_histogram = hist.view() / np.max(hist.view())
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(
                other_histogram > 0, reference_histogram / other_histogram, -10
            )
        weights = weights / np.max(weights)

        weights[weights < 0] = 1
        weights[weights == np.nan] = 1

        weights_list[key] = weights
    return weights_list
