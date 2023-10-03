import gc
import unittest

import torch

from diffusers import (
    TortoiseTTSPipeline,
)
from diffusers.utils.testing_utils import enable_full_determinism, slow

from ..pipeline_params import TEXT_TO_AUDIO_BATCH_PARAMS, TEXT_TO_AUDIO_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class TortoiseTTSPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = TortoiseTTSPipeline
    # TODO: copied from AudioLDMPipelineFastTests, may need to be changed
    params = TEXT_TO_AUDIO_PARAMS
    batch_params = TEXT_TO_AUDIO_BATCH_PARAMS
    required_optional_params = frozenset(
        [
            "num_inference_steps",
            "num_waveforms_per_prompt",
            "generator",
            "latents",
            "output_type",
            "return_dict",
            "callback",
            "callback_steps",
        ]
    )

    def get_dummy_components(self):
        pass

    def get_dummy_inputs(self, device, seed=0):
        pass


@slow
class TortoiseTTSPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self):
        pass
