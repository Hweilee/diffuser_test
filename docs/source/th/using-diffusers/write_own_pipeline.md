<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# คำเกริ่นนำ

ยินดีต้อนรับสู่ 🧨 Diffusers! หากคุณเป็นมือใหม่ในโมเดลการแพร่กระจายและ AI ที่สร้างสรรค์ และต้องการเรียนรู้เพิ่มเติม คุณมาถูกที่แล้ว บทเรียนเหล่านี้ที่เข้าใจง่ายสำหรับผู้เริ่มต้นถูกออกแบบมาเพื่อให้เป็นการแนะนำอย่างอ่อนโยนในโมเดลการแพร่กระจาย และช่วยให้คุณเข้าใจหลักการของไลบรารี - ส่วนประกอบหลักและวิธีการใช้ 🧨 Diffusers

คุณจะเรียนรู้ว่าจะใช้แบบจำลองและตัวตั้งเวลาเพื่อประกอบระบบการแพร่กระจายสำหรับการทำนาย โดยเริ่มต้นด้วยไพป์ไลน์ขั้นพื้นฐานแล้วค่อย ๆ ก้าวไปสู่ไพป์ไลน์ Stable Diffusion.

## แยกส่วนของไพป์ไลน์พื้นฐาน

ไพป์ไลน์เป็นวิธีที่รวดเร็วและง่ายในการเรียกใช้โมเดลสำหรับการทำนาย โดยไม่จำเป็นต้องมีมากกว่าสี่บรรทัดโค้ดเพื่อสร้างภาพ:

```py
>>> from diffusers import DDPMPipeline

>>> ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
>>> image = ddpm(num_inference_steps=25).images[0]
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ddpm-cat.png" alt="Image of cat created from DDPMPipeline"/>
</div>

แต่ไพป์ไลน์ทำงานอย่างไร? มาแยกการทำงานของไพป์ไลน์และดูว่ามีอะไรเกิดขึ้นภายใน

ในตัวอย่างข้างบนนี้ ไพป์ไลน์มีโครงสร้าง [`UNet2DModel`] และ [`DDPMScheduler`] ไพป์ไลน์ทำการลด noise ในภาพโดยใช้รูปแบบ random noise กับขนาดของผลลัพธ์ที่ต้องการและส่งผ่านมันผ่านโมเดลหลายๆ ครั้ง ในแต่ละ timestep โดยทุกๆ timestep โมเดลทำการทำนาย *noisy residuals* และ schedulers ใช้มันเพื่อทำการทำนายภาพที่ไม่มี noise ไพป์ไลน์ทำการทำซ้ำกระบวนการนี้จนกว่าจะถึงจุดสิ้นสุดของจำนวนขั้นตอนทำนายที่ระบุไว้

เพื่อสร้างไพป์ไลน์นี้อีกครั้ง ด้วยการใช้โมเดลและ schedulers โดยแยกกัน มาเขียนกระบวนการ diffusion ของเราเอง.

1. โหลดโมเดลและ schedulers:

```py
>>> from diffusers import DDPMScheduler, UNet2DModel

>>> scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
>>> model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
```

2. กำหนดจำนวนขั้นตอนเวลาที่จะทำกระบวนการ denoise:

```py
>>> scheduler.set_timesteps(50)
```

3. การตั้งค่า timesteps จะสร้างเทนเซอร์ที่มีองค์ประกอบที่กระจายเป็นเท่ากัน 50 ตัวอย่างในที่นี้ แต่ละองค์ประกอบมีค่าที่สอดคล้องกับเวลาที่โมเดลทำนายภาพที่ไม่มี noise การทำกระบวนการลด noise ภายหลังคุณจะทำการวนรอบเหนือเทนเซอร์นี้เพื่อทำการลด noise ในภาพ:

```py
>>> scheduler.timesteps
tensor([980, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740, 720,
    700, 680, 660, 640, 620, 600, 580, 560, 540, 520, 500, 480, 460, 440,
    420, 400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160,
    140, 120, 100,  80,  60,  40,  20,   0])
```

4. สร้าง random noise ที่มีขนาดเดียวกับผลลัพธ์ที่ต้องการ:

```py
>>> import torch

>>> sample_size = model.config.sample_size
>>> noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")
```

5. ต่อไป เขียนการวนรอบเพื่อวนลูปในขั้นตอนเวลา ที่แต่ละ timestep โมเดลทำ [`UNet2DModel.forward`] และส่งคืน เมธอด [`~DDPMScheduler.step`] ของ scheduler นำ previous sample เป็นที่ผ่านมา จะใช้เป็นป้อนอินพุทต่อไปให้โมเดลในลูปการลด noise และก็จะทำการทำซ้ำจนกว่าจะถึงจุดสิ้นสุดของอาร์เรย์ `timesteps`:

```py
>>> input = noise

>>> for t in scheduler.timesteps:
...     with torch.no_grad():
...         noisy_residual = model(input, t).sample
...     previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
...     input = previous_noisy_sample
```

นี้คือกระบวนการลด noise ทั้งหมด และคุณสามารถใช้รูปแบบเดียวกันนี้เพื่อเขียนระบบการแพร่กระจายใดๆ

6. ขั้นตอนสุดท้ายคือแปลงเอาต์พุทที่ถูกลด noise ให้เป็นภาพ:

```py
>>> from PIL import Image
>>> import numpy as np

>>> image = (input / 2 + 0.5).clamp(0, 1).squeeze()
>>> image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
>>> image = Image.fromarray(image)
>>> image
```

ในส่วนต่อไป คุณจะทดสอบทักษะของคุณและแยกส่วนไพป์ไลน์ Stable Diffusion ที่ซับซ้อนขึ้น ขั้นตอนมีมากน้อยคล้ายกัน คุณจะเริ่มต้นด้วยการสร้างส่วนประกอบที่จำเป็น และตั้งค่าจำนวนขั้นตอนเวลาเพื่อสร้างอาร์เรย์ `timestep` อาเรย์ `timestep` นี้ใช้ในการวนลูปการลด noise และสำหรับแต่ละองค์ประกอบในอาร์เรย์นี้ โมเดลทำนายภาพที่ไม่มี noise  การทำนายเหล่านี้ทำการวนลูปที่ `timestep` และในแต่ละ `timestep` มันจะส่งอาร์เรย์ที่เป็นต้นฉบับและ scheduler จะใช้มันในการทำนายภาพที่ไม่มี noise เพื่อเป็นอินพุทของโมเดลในลูปและมันจะทำการทำซ้ำจนกว่าจะถึงจุดสิ้นสุดของอาร์เรย์ `timestep`

ลองมันออก!

## แยกส่วนไพป์ไลน์ Stable Diffusion

Stable Diffusion เป็นโมเดล *latent diffusion* มันเรียกว่าโมเดล latent diffusion เนื่องจากมันทำงานกับการแสดงบทบาทที่มีมิติต่ำของรูปภาพแทนที่จะเป็นพื้นที่พิกเซลจริง ซึ่งทำให้มีประสิทธิภาพในการจัดเก็บหน่วยความจำมากขึ้น ตัวเข้ารหัสบีบอัดรูปภาพเป็นการแทนที่เล็กลง และตัวถอดรหัสเพื่อแปลงรายการที่บีบอัดให้กลับเป็นรูปภาพ สำหรับโมเดลการแปลนข้อความไปยังรูปภาพ คุณต้องใช้ตัวเข้ารหัสและตัวถอดรหัสเพื่อสร้างตัวฝังข้อความ จากตัวอย่างก่อนหน้านี้ คุณคงรู้ว่าคุณต้องใช้โมเดล UNet และ schedulers

เมื่อคุณทราบว่าคุณต้องการอะไรสำหรับไพป์ไลน์ Stable Diffusion โหลดส่วนประกอบทั้งหมดเหล่านี้ด้วยเมธอด [`~ModelMixin.from_pretrained`]:

```py
>>> from PIL import Image
>>> import torch
>>> from transformers import CLIPTextModel, CLIPTokenizer
>>> from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

>>> vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
>>> tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
>>> text_encoder = CLIPTextModel.from_pretrained(
...     "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
... )
>>> unet = UNet2DConditionModel.from_pretrained(
...     "CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True
... )
```

และใช้ [`UniPCMultistepScheduler`] แทน [`PNDMScheduler`] เพื่อให้เห็นว่ามันง่ายแค่ไหนที่จะใส่ตัว scheduler ที่แตกต่าง:

```py
>>> from diffusers import UniPCMultistepScheduler

>>> scheduler = UniPCMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
```

เพื่อเร่งความเร็วในการทำนาย ย้ายโมเดลไปยัง GPU:

```py
>>> torch_device = "cuda"
>>> vae.to(torch_device)
>>> text_encoder.to(torch_device)
>>> unet.to(torch_device)
```

### สร้าง tokenizer

ขั้นตอนต่อไปคือการแปลงข้อความเป็น embedding ข้อความใช้เพื่อแสดงบทบาทของ UNet และนำทางกระบวนการลด noise ไปสู่บางสิ่งที่คล้ายกับข้อมูลป้อนของคุณ

<Tip>

💡 พารามิเตอร์ `guidance_scale` กำหนดว่าต้องให้น้ำหนักมากเพียงใดกับ prompt เมื่อสร้างภาพ

</Tip>

อาจเลือกใช้ prompt ใดก็ได้ตามที่คุณต้องการที่จะสร้างสิ่งใด!

```py
>>> prompt = ["a photograph of an astronaut riding a horse"]
>>> height = 512  # default height of Stable Diffusion
>>> width = 512  # default width of Stable Diffusion
>>> num_inference_steps = 25  # Number of denoising steps
>>> guidance_scale = 7.5  # Scale for classifier-free guidance
>>> generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise
>>> batch_size = len(prompt)
```

Tokenize ข้อความและสร้าง embedding ข้อความจากคำเสนอ:

```py
>>> text_input = tokenizer(
...     prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
... )

>>> with torch.no_grad():
...     text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
```

สร้าง *unconditional embedding*

```py
>>> max_length = text_input.input_ids.shape[-1]
>>> uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
>>> uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
```

รวม embedding ที่มีเงื่อนไขและที่ไม่มีเงื่อนไขเข้าด้วยกัน:

```py
>>> text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
```

### สร้าง random noise

ต่อไป สร้าง random noise เพื่อเป็นจุดเริ่มต้นของกระบวนการ diffusion

<Tip>

💡 ความสูงและความกว้างต้องถูกหารด้วย 8 ได้เพราะโมเดล `vae` คุณสามารถตรวจสอบได้โดยรันโค้ดนี้:

```py
2 ** (len(vae.config.block_out_channels) - 1) == 8
```

</Tip>

```py
>>> latents = torch.randn(
...     (batch_size, unet.in_channels, height // 8, width // 8),
...     generator=generator,
... )
>>> latents = latents.to(torch_device)
```

### Denoising

เริ่มต้นด้วยการปรับอินพุทด้วย *ซิกม่า* ซึ่งเป็นที่จำเป็นสำหรับ scheduler บางตัวเช่น [`UniPCMultistepScheduler`] :

```py
>>> latents = latents * scheduler.init_noise_sigma
```

ขั้นสุดท้ายคือการสร้าง denoising loop จำไว้ว่าลูปต้องทำสามสิ่งนี้:

1. ตั้งค่า scheduler
2. เริ่มวนลูปในแต่ละ timestep
3. ที่แต่ละ timestep เรียกโมเดล UNet เพื่อทำนาย noisy latents ที่เหลือเพื่อให้ได้ภาพที่ไม่มี noise

```py
>>> from tqdm.auto import tqdm

>>> scheduler.set_timesteps(num_inference_steps)

>>> for t in tqdm(scheduler.timesteps):
...     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
...     latent_model_input = torch.cat([latents] * 2)

...     latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

...     # predict the noise residual
...     with torch.no_grad():
...         noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

...     # perform guidance
...     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
...     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

...     # compute the previous noisy sample x_t -> x_t-1
...     latents = scheduler.step(noise_pred, t, latents).prev_sample
```

### ถอดรหัสภาพ

ขั้นสุดท้ายคือการใช้ `vae` เพื่อถอดรหัส latents ให้กลายเป็นภาพและดึงเอาออกมาด้วย `sample`:

```py
# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample
```

สุดท้าย แปลงภาพเป็น `PIL.Image` เพื่อดูภาพที่สร้างขึ้น!

```py
>>> image = (image / 2 + 0.5).clamp(0, 1).squeeze()
>>> image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
>>> images = (image * 

255).round().astype("uint8")
>>> image = Image.fromarray(image)
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/blog/assets/98_stable_diffusion/stable_diffusion_k_lms.png"/>
</div>

## ขั้นตอนต่อไป
* เรียนรู้วิธีการ [สร้างและเป็นส่วนหนึ่งของการพัฒนาไพป์ไลน์](contribute_pipeline) ใน 🧨 Diffusers!
* สำรวจ [ไพป์ไลน์ที่มีอยู่](../api/pipelines/overview) ในไลบรารี และดูว่าคุณสามารถแยกส่วนเพื่อสร้างไพป์ไลน์ใหม่ๆ ได้หรือไม่