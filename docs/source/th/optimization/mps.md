<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Metal Performance Shaders (MPS)

🤗 Diffusers รองรับการใช้งานกับ Apple silicon (ชิป M1/M2) โดยใช้ [`mps`](https://pytorch.org/docs/stable/notes/mps.html) device ใน PyTorch ซึ่งใช้ Metal framework เพื่อให้การใช้ GPU บนอุปกรณ์ MacOS ได้เต็มประสิทธิภาพ คุณจำเป็นต้องมี:

- คอมพิวเตอร์ macOS ที่มีฮาร์ดแวร์ Apple silicon (M1/M2)
- macOS เวอร์ชัน 12.6 หรือใหม่กว่า (แนะนำให้ใช้ 13.0 หรือใหม่กว่า)
- เวอร์ชัน arm64 ของ Python
- [PyTorch 2.0](https://pytorch.org/get-started/locally/) (แนะนำ) หรือ 1.13 (เวอร์ชันขั้นต่ำที่รองรับสำหรับ `mps`)

Backend `mps` ใช้ PyTorch's `.to()` interface เพื่อย้าย Stable Diffusion pipeline ไปยังอุปกรณ์ M1 หรือ M2 ของคุณ:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# แนะนำให้ทำเมื่อคอมพิวเตอร์ของคุณมี RAM น้อยกว่า 64 GB
pipe.enable_attention_slicing()

prompt = "a photo of an astronaut riding a horse on mars"
```

<Tip warning={true}>

การสร้าง prompt หลาย ๆ รายการในชุดอาจ [crash](https://github.com/huggingface/diffusers/issues/363) หรือไม่ทำงานได้อย่างเสถียรภาพ นี้เชื่อว่าเกี่ยวข้องกับ backend [`mps`](https://github.com/pytorch/pytorch/issues/84039) ใน PyTorch ในขณะที่กำลังได้รับการสำรวจ ควรทำการทดลองแทนการทำงานเป็นกลุ่ม

</Tip>

หากคุณใช้ **PyTorch 1.13**, คุณต้อง "prime" pipeline ด้วยการผ่าน pass อีกครั้งเพียงครั้งเดียว นี้เป็นการแก้ปัญหาชั่วคราวสำหรับปัญหาที่ผ่านการ infer ครั้งแรกมีผลลัพธ์ที่แตกต่างเล็กน้อยจากการ infer ต่อมา คุณเพียงต้องทำการผ่านนี้เพียงครั้งเดียว และหลังจากเพียงเพียงข้ามไปผ่านอินเฟอร์เซ็นเพียงครั้งเดียว คุณสามารถทิ้งผลลัพธ์ได้.

```diff
  from diffusers import DiffusionPipeline

  pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("mps")
  pipe.enable_attention_slicing()

  prompt = "a photo of an astronaut riding a horse on mars"
# First-time "warmup" pass if PyTorch version is 1.13
+ _ = pipe(prompt, num_inference_steps=1)

# ผลลัพธ์ตรงกับที่มีจากอุปกรณ์ CPU หลังจากการผ่าน warmup
  image = pipe(prompt).images[0]
```

## แก้ปัญหา

Performance ของ M1/M2 ถูกบีบอัดอย่างมากต่อความดันของหน่วยความจำ กรณีนี้เกิดขึ้น ระบบจะทำการสลับอัตโนมัติตามความจำเมื่อจำเป็นซึ่งทำให้ปรสformanceถดถอยอย่างมีนัยสำคัญ

เพื่อป้องกันปัญหานี้ แนะนำให้ใช้ *attention slicing* เพื่อลดความดันของหน่วยความจำในขณะที่ infer และป้องกันการสลับ นี้มีความเกี่ยวข้องมากนับเนื่องจากระบบของคุณมี RAM น้อยกว่า 64GB หรือหากคุณสร้างภาพที่มีขนาดที่ไม่มีมาตรฐานมากกว่า 512×512 พิกเซล ใช้ [`~DiffusionPipeline.enable_attention_slicing`] function บน pipeline ของคุณ:

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
pipeline.enable_attention_slicing()
```

Attention slicing ทำการดำเนินการ attention 

ที่ใช้ค่าใช้จ่ายสูงในหลาย ๆ ขั้นตอนแทนที่จะทำทั้งหมดในครั้งเดียว มักทำให้ Performance ดีขึ้นประมาณ ~20% ในคอมพิวเตอร์ที่ไม่มี universal memory แต่เราตรวจสอบ *Performance* ในคอมพิวเตอร์ Apple silicon ส่วนใหญ่ ยกเว้นหากคุณมี RAM 64GB หรือมากกว่า.