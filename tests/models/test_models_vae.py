# coding=utf-8
# Copyright 2022 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from diffusers import AutoencoderKL
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import floats_tensor, slow, torch_device
from parameterized import parameterized

from ..test_modeling_common import ModelTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class AutoencoderKLTests(ModelTesterMixin, unittest.TestCase):
    model_class = AutoencoderKL

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)

        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 4,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_training(self):
        pass

    def test_from_pretrained_hub(self):
        model, loading_info = AutoencoderKL.from_pretrained("fusing/autoencoder-kl-dummy", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = AutoencoderKL.from_pretrained("fusing/autoencoder-kl-dummy")
        model = model.to(torch_device)
        model.eval()

        # One-time warmup pass (see #372)
        if torch_device == "mps" and isinstance(model, ModelMixin):
            image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
            image = image.to(torch_device)
            with torch.no_grad():
                _ = model(image, sample_posterior=True).sample
            generator = torch.manual_seed(0)
        else:
            generator = torch.Generator(device=torch_device).manual_seed(0)

        image = torch.randn(
            1,
            model.config.in_channels,
            model.config.sample_size,
            model.config.sample_size,
            generator=torch.manual_seed(0),
        )
        image = image.to(torch_device)
        with torch.no_grad():
            output = model(image, sample_posterior=True, generator=generator).sample

        output_slice = output[0, -1, -3:, -3:].flatten().cpu()

        # Since the VAE Gaussian prior's generator is seeded on the appropriate device,
        # the expected output slices are not the same for CPU and GPU.
        if torch_device == "mps":
            expected_output_slice = torch.tensor(
                [
                    -4.0078e-01,
                    -3.8323e-04,
                    -1.2681e-01,
                    -1.1462e-01,
                    2.0095e-01,
                    1.0893e-01,
                    -8.8247e-02,
                    -3.0361e-01,
                    -9.8644e-03,
                ]
            )
        elif torch_device == "cpu":
            expected_output_slice = torch.tensor(
                [-0.1352, 0.0878, 0.0419, -0.0818, -0.1069, 0.0688, -0.1458, -0.4446, -0.0026]
            )
        else:
            expected_output_slice = torch.tensor(
                [-0.2421, 0.4642, 0.2507, -0.0438, 0.0682, 0.3160, -0.2018, -0.0727, 0.2485]
            )

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))


@slow
class AutoencoderKLIntegrationTests(unittest.TestCase):
    def get_sd_image(self, seed=0):
        batch_size = 4
        channels = 3
        height = 512
        width = 512
        image = torch.randn(batch_size, channels, height, width, device=torch_device)
        return image

    def get_sd_vae_model(self, fp16=False):
        revision = "fp16" if fp16 else None
        torch_dtype = torch.float16 if fp16 else torch.float32

        model = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch_dtype, revision=revision
        )
        model.to(torch_device).eval()

        return model

    def get_generator(self, seed=0):
        return torch.Generator(device=torch_device).manual_seed(seed)

    def test_stable_diffusion(self):
        pass

    def test_stable_diffusion_fp16(self):
        pass

    def test_stable_diffusion_mode(self):
        pass

    def test_stable_diffusion_decode(self):
        pass

    def test_stable_diffusion_decode_fp16(self):
        pass

    @parameterized.expand(
        [
            # fmt: off
            [33, [-7.3992, -2.9123, -2.8571,  0.5112, -4.9118, -0.0620, -5.7185,  3.4767, -1.7884]],
            [47, [-3.5498, -5.4690, -6.9103, -4.4613, -2.6730,  0.6610,  0.0283, -5.7788, -1.9802]],
            # fmt: on
        ]
    )
    def test_stable_diffusion_encode_sample(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)

        with torch.no_grad():
            dist = model.encode(image).latent_dist
            sample = dist.sample(generator=generator)

        assert sample.shape == [image.shape[0], 4] + [i // 8 for i in image.shape[2:]]

        output_slice = sample[0, -1, -3:, -3:].flatten().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, atol=1e-4))
