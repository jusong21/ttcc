from functools import partial

# for dileptonic analyzer
from BTVNanoCommissioning.workflows.ttcc2L2Nu_producer import (
    NanoProcessor as TTCCdilepProcessor,
)
# for TTCC ntuplizer
from BTVNanoCommissioning.workflows.ttcc_ntuplizer import (
    NanoProcessor as TTCCntuplizer,
)

# FIXME - make names more systematic?
workflows = {}

workflows["ttcc2L2Nu"] = TTCCdilepProcessor
workflows["ttccNtuple"] = TTCCntuplizer

__all__ = ["workflows"]
