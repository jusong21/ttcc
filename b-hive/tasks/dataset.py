import os
import random
from typing import Dict
from collections import defaultdict
import uuid
import hist
import luigi
import numpy as np
from coffea import processor
from coffea.nanoevents import BaseSchema
from rich.progress import track

from tasks.base import BaseTask
from tasks.parameter_mixins import DatasetDependency
from utils.coffea_processors.processor_loader import ProcessorLoader
from utils.config.config_loader import ConfigLoader
from utils.dataset.merging import merge_datasets
from utils.weighting.histogram import (
    histogram_weighting,
    weight_all_files_histrogram_weighting,
)


def read_in_samples_match_processes(file_path, processes):
    samples_dict = defaultdict(list)
    with open(file_path, "r") as input_txt:
        while line := input_txt.readline().rstrip("\n"):
            if line:
                if processes == ["default"]:
                    samples_dict[uuid.uuid4().hex].append(line)
                else:
                    for process in processes:
                        if process in line:
                            samples_dict[process].append(line)
                            break
    return samples_dict


class DatasetConstructorTask(DatasetDependency, BaseTask):
    coffea_worker = luigi.IntParameter(
        default=1,
        description="Number of workers for Coffea-processing",
        significant=False,
    )

    chunk_size = luigi.IntParameter(
        default=100000, description="Number of events for outgoing files"
    )

    def output(self):
        return {
            "file_list": self.local_target("processed_files.txt"),
            "histogram": self.local_target("histogram.npy"),
        }

    def run(self):
        print("Dataset construction")
        self.output()["file_list"].parent.touch()  # create directory
        config = ConfigLoader.load_config(self.config)
        np.random.seed(self.seed)
        assert (
            self.filelist != "",
            """You did not specify a filelist .txt but tried to run a new DatasetConstruction!
Either you forgot to specify the path to the file or are using a wrong dataset-version!""",
        )

        all_files = []
        samples = read_in_samples_match_processes(self.filelist, config["processes"])

        if self.debug:
            # skim files to only 10 files per process
            samples = {
                key: values[0 : min(len(samples), 1)] for key, values in samples.items()
            }

        # Make a dictionary entry for all of them:

        futures_run = processor.Runner(
            executor=processor.FuturesExecutor(
                compression=None, workers=self.coffea_worker
            ),
            schema=BaseSchema,
            chunksize=self.chunk_size//20, # should be << chunk_size in order to get everything shuffled correctly
            maxchunks=None if not (self.debug) else 10,
        )
        #processorClass = ProcessorLoader(config.get("processor", "PFCandidateAndVertexProcessing"),  
        processorClass = ProcessorLoader(config.get("processor", "TTCCLZ4Processing"),  
                output_directory=self.local_path(),
                #bins_pt=config.get("bins_pt", None),
                #bins_eta=config.get("bins_eta", None),
                #bins_class=config.get("bins_class", None),
                bins_nbjets=config.get("bins_nbjets", None),
                bins_ncjets=config.get("bins_ncjets", None),
                processes=config.get("processes", None),
                global_features=config.get("global_features", []),
                jet_features=config.get("jet_features", []),
                lepton_features=config.get("lepton_features", []),
                n_jet_candidates=config.get("n_jet_candidates", 4),
                n_lepton_candidates=config.get("n_lepton_candidates", 2),
                truths=config.get("truths", None),
                )
        print("Processor:")
        print(processorClass)

        output = futures_run(
            samples,
            treename=config["treename"],
            processor_instance=processorClass
        )

        # saving histograms from coffea
        histograms = []
        file_list = []
        for key, value in output.items():
            if key == "output_location":
                file_list += value  # append output location list
            else:
                histograms.append(value.view())  # append view on hist -> np.array

        training_histograms = {
            key: value for key, value in output.items() if not "output_location" in key
        }
        #print('train _hist')
        #for key in training_histograms:
        #    print(key)
        #    print(training_histograms[key])

        np.save(
            self.output()[f"histogram"].path,
            np.array(histograms, dtype=np.float32),
        )
        print(f"number of output files:\t", len(file_list))

        print("Start merging files")
        # returns list of merged training-files
        all_files += merge_datasets(
            file_list,
            self.local_path(),
            label="file",
            #processor=config.get("processor", "PFCandidateAndVertexProcessing"),
            processor=config.get("processor", "TTCCLZ4Processing"),
            chunk_size=self.chunk_size,
            shuffle=True,
            histograms=np.array(histograms, dtype=np.float32),
            #reference_key=0,
            reference_key=0,
            #bins_pt=config["bins_pt"],
            #bins_eta=config["bins_eta"],
            bins_nbjets=config["bins_nbjets"],
            bins_ncjets=config["bins_ncjets"],
        )
        if "LZ4" not in config.get("processor", "TTCCLZ4Processing"):
            # delete unmerged files
            for file in file_list:
                os.remove(file)

            # add weights to all files
            # this should be done on the fly - please implement!
            all_files = weight_all_files_histrogram_weighting(
                all_files,
                histograms=training_histograms,
                #bins_pt=config["bins_pt"],
                #bins_eta=config["bins_eta"],
                bins_nbjets=config["bins_nbjets"],
                bins_ncjets=config["bins_ncjets"],
                reference_key=config["reference_flavour"],
            )

        self.output()["file_list"].dump("\n".join(all_files), formatter="text")
