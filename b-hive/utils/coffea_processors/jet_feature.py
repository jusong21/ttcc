import awkward as ak
import numpy as np
from functools import reduce

from utils.coffea_processors.base import DataPreprocessing_BaseClass
from utils.dataset.structured_arrays import (
    structured_array_from_tree,
    structured_array_from_tree_truth_from_dict,
)


class JetFeatureProcessing(DataPreprocessing_BaseClass):
    def callColumnAccumulator(self, output, events, flag, **kwargs):
        # slicing based on p_T and eta

        if isinstance(self.truths, dict):
            truth_arr = structured_array_from_tree_truth_from_dict(
                events=events,
                truth_dict=self.truths,
                precision=np.bool8,
                feature_length=1,
            )
        else:
            truth_arr = structured_array_from_tree(
                events=events,
                keys=self.truths,
                precision=np.bool8,
                feature_length=1,
            )

        global_arr = structured_array_from_tree(
            #events=events[data_slice],
            events=events,
            keys=self.global_features,
            precision=self.precision,
            feature_length=1,
        )

        jet_arr = structured_array_from_tree(
            #events=events[data_slice],
            events=events,
            keys=self.jet_features,
            precision=self.precision,
            feature_length=self.n_jet,
        )

        # create an array with the process value
        process = np.ones(len(global_arr)) * flag

        glob_mask = reduce(
            np.logical_and,
            [~np.any(np.isnan(global_arr[key])) for key in global_arr.dtype.names],
        )
        if jet_arr.dtype.names:
            jet_mask = reduce(
                np.logical_and,
                [
                    ~np.any(np.isnan(jet_arr[key]), axis=1)
                    for key in jet_arr.dtype.names
                ],
            )
        else:
            jet_mask = np.ones(len(global_arr))

        nan_mask = reduce(np.logical_and, [glob_mask, jet_mask])

        return (
            global_arr[nan_mask],
            jet_arr[nan_mask],
            truth_arr[nan_mask],
            process[nan_mask],
        )

    def saveOutput(
        self, output_location, global_arr, jet_arr, truth, process
    ):
        np.savez(
            output_location,
            global_features=global_arr,
            jet_features=jet_arr,
            truth=truth,
            process=process,
        )
