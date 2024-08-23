from utils.coffea_processors.pf_candidate_and_vertex import PFCandidateAndVertexProcessing
from utils.coffea_processors.lz4_processor import LZ4Processing 
from utils.coffea_processors.lz4fp16_processor import LZ4FP16Processing 
from utils.coffea_processors.l1_processor import L1PFCandidateAndVertexProcessing 
from utils.coffea_processors.ttcclz4_processor import TTCCLZ4Processing

class ProcessorClasses:
    TTCCLZ4Processing = "TTCCLZ4Processing" 
    PFCandidateAndVertexProcessing = "PFCandidateAndVertexProcessing" 
    LZ4Processing = "LZ4Processing" 
    LZ4FP16Processing = "LZ4FP16Processing" 
    L1PFCandidateAndVertexProcessing = "L1PFCandidateAndVertexProcessing"


def ProcessorLoader(model: str = "", *args, **kwargs):
    match model:
        case ProcessorClasses.TTCCLZ4Processing:
            return TTCCLZ4Processing(*args, **kwargs)
        case ProcessorClasses.PFCandidateAndVertexProcessing:
            return PFCandidateAndVertexProcessing(*args, **kwargs)
        case ProcessorClasses.LZ4Processing:
            return LZ4Processing(*args, **kwargs)
        case ProcessorClasses.LZ4FP16Processing:
            return LZ4FP16Processing(*args, **kwargs)
        case ProcessorClasses.L1PFCandidateAndVertexProcessing:
            return L1PFCandidateAndVertexProcessing(*args, **kwargs)
        case _:
            return PFCandidateAndVertexProcessing(*args, **kwargs) 
