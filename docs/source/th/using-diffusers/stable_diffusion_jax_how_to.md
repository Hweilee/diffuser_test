# JAX/Flax

[[open-in-colab]]

🤗 Diffusers รองรับ Flax สำหรับการสร้างผลลัพธ์อย่างรวดเร็วบน Google TPUs เช่น ที่มีให้ใน Colab, Kaggle หรือ Google Cloud Platform คู่มือนี้จะแนะนำวิธีการทำ Inference ด้วย Stable Diffusion โดยใช้ JAX/Flax.

ก่อนที่คุณจะเริ่มต้น, ตรวจสอบให้แน่ใจว่าคุณได้ติดตั้งไลบรารีที่จำเป็น:

```py
# uncomment to install the necessary libraries in Colab
#!pip install -q jax==0.3.25 jaxlib==0.3.25 flax transformers ftfy
#!pip install -q diffusers
```

คุณควรตรวจสอบให้แน่ใจว่าคุณกำลังใช้ TPU backend ด้วย โดยทั่วไป JAX ไม่ทำงานอย่างเพียงที่ TPUs แต่คุณจะได้รับประสิทธิภาพที่ดีที่สุดใน TPU เพราะทุกเซิร์ฟเวอร์มี TPU accelerators 8 ตัวทำงานพร้อมกัน.

หากคุณกำลังทำงานคำแนะนำนี้ใน Colab, เลือก *Runtime* ในเมนูด้านบน, เลือก *Change runtime type*, แล้วเลือก *TPU* ภายใต้การตั้งค่า *Hardware accelerator*. Import JAX และตรวจสอบทันทีว่าคุณกำลังใช้ TPU:

```python
import jax
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()

num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")
assert (
    "TPU" in device_type, 
    "Available device is not a TPU, please select TPU from Edit > Notebook settings > Hardware accelerator"
)
"Found 8 JAX devices of type Cloud TPU."
```

เยี่ยม, ตอนนี้คุณสามารถ import ส่วนที่เหลือของ dependencies ที่คุณจำเป็น:

```python
import numpy as np
import jax.numpy as jnp

from pathlib import Path
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image

from huggingface_hub import notebook_login
from diffusers import FlaxStableDiffusionPipeline
```

## Load a model

Flax เป็น functional framework, ดังนั้นโมเดลไม่มี state และพารามิเตอร์ถูกเก็บนอกไปจากนั้น โหลด pretrained Flax pipeline จะได้ทั้ง *pipeline* และน้ำหนักของโมเดล (หรือพารามิเตอร์) ในคู่มือนี้, คุณจะใช้ `bfloat16`, ประเภท half-float ที่มีประสิทธิภาพมากขึ้นที่ได้รับการสนับสนุนจาก TPUs (คุณสามารถใช้ `float32` สำหรับความแม่นยำเต็มถ้วนได้ถ้าคุณต้องการ).

```python
dtype = jnp.bfloat16
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="bf16",
    dtype=dtype,
)
```

## Inference

TPUs มักมี 8 devices ทำงานพร้อมกัน, ดังนั้นให้ใช้ prompt เดียวกันสำหรับทุก device. นี้หมายความว่าคุณสามารถทำ inference บน 8 devices พร้อมกัน, โดยที่แต่ละ device สร้างภาพหนึ่งภาพ ดังนั้นคุณจะได้รับ 8 ภาพในเวลาเดียวกันที่ใช้ในการสร้างภาพเดียว!

<Tip>

เรียนรู้รายละเอียดเพิ่มเติมใน [How does parallelization work?](#how-does-parallelization-work) section.

</Tip>

หลังจากทำการ replicate คำใบ้, รับ tokenized text ids โดยเรียกใช้ฟังก์ชัน `prepare_inputs` บน pipeline. ความยาวของ tokenized text ถูกตั้งค่าเป็น 77 tokens ตามที่ต้องการโดยการกำหนดค่าในการกำหนดค่าด้านล่างของโมเดล CLIP text.

```python
prompt = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
prompt = [prompt] * jax.device_count()
prompt_ids = pipeline.prepare_inputs(prompt)
prompt_ids.shape
"(8, 77)"
```

พารามิเตอร์และข้อมูลนำเข้าต้องถูก replicate ไปยัง 8 parallel devices. พารามิเตอร์จะถูก replicate ด้วย [`flax.jax_utils.replicate`](https://flax.readthedocs.io/en/latest/api_reference/flax.jax_utils.html#flax.jax_utils.replicate) ซึ่งวางแผนที่จะเปลี่ยนรูปร่างของน้ำหนักเพื่อให้ทำซ้ำ 8 ครั้ง. ข้อมูลเข้าจะถูก replicate โดยใช้ `shard`.

```python
# parameters
p_params = replicate(params)

# arrays
prompt_ids = shard(prompt_ids)
prompt_ids.shape
"(8, 1, 77)"
```

รูปร่างนี้หมายถึงทุก ๆ อุปกรณ์ 8 รับข้อมูลเข้าในรูปแบบ `jnp` array ที่มีรูปร่าง `(1, 77)`, โดยที่ `1` คือขนาด batch ต่ออุปกรณ์. ใน TPUs ที่มีหน่วยความจำเพียงพอ, คุณสามารถมีขนาด batch มากกว่า `1` หากคุณต้องการสร้างภาพหลายภาพ (ต่อ chip) พร้อมกัน.

ต่อมา, สร้าง random number generator เพื่อส่งไปยังฟังก์ชันการสร้าง. นี้เป็นขั้นตอนมาตรฐานใน Flax ที่เป็นเรื่องจริงจังและมีความคิดเห็นเฉียบพลันเกี่ยวกับเรขาคณิตสุ่ม. ฟังก์ชันช่วยด้านล่างใช้ seed เพื่อเริ่มต้น random number generator. ในขณะที่คุณใช้ seed เดียวกัน, คุณจะได้ผลลัพธ์เหมือนกันเสมอ. คุณสามารถใช้ seed ต่างกันเมื่อคุณสำรวจผลลัพธ์ในภายหลังในคู่มือนี้.

```python
def create_key(seed=0):
    return jax.random.PRNGKey(seed)
```

ฟังก์ชันช่วยหรือ `rng` ถูก split 8 ครั้งเพื่อให้แต่ละอุปกรณ์ได้รับ generator ที่แตกต่างกันและสร้างภาพที่แตกต่างกัน.

```python
rng = create_key(0)
rng = jax.random.split(rng, jax.device_count())
```

เพื่อให้ใช้ประโยชน์จากความเร็วที่ถูกปรับให้เหมาะกับ TPU ให้ส่ง `jit=True` ไปยัง pipeline เพื่อคอมไพล์โค้ด JAX เป็นการแทนที่ทำให้โมเดลทำงานแบบ parallel บน 8 อุปกรณ์.

<Tip warning={true}>

คุณต้องให้แน่ใจว่าข้อมูลเข้าของคุณมีรูปร่างเดียวกันในการเรียกใช้ต่อๆ กัน, ไม่เช่นนั้น JAX จะต้องทำการคอมไพล์โค้ดใหม่ซึ่งช้าขึ้น.

</Tip>

การรันการทำนายครั้งแรกใช้เวลามากขึ้นเนื่องจากต้องคอมไพล์โค้ด, แต่การเรียกใช้ซ้ำ (แม้ด้วยข้อมูลนำเข้าที่แตกต่างกัน) จะเร็วมาก ตัวอย่างเช่น, ใน TPU v2-8, ใช้เวลามากกว่า **7s** ในการเรียกใช้เพื่อนรุ่นใหม่!

```py
%%time
images = pipeline(prompt_ids, p_params, rng, jit=True)[0]

"CPU times: user 56.2 s, sys: 42.5 s, total: 1min 38s"
"Wall time: 1min 29s"
```

array ที่ได้มีรูปร่าง `(8, 1, 512, 512, 3)` ซึ่งควรถูก reshape เพื่อลบมิติที่สองและได้ 8 ภาพขนาด `512 × 512 × 3`. จากนั้นคุณสามารถใช้ [`~utils.numpy_to_pil`] เพื่อแปลง array เป็นรูปภาพ.

```python
from diffusers import make_image_grid

images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)
make_image_grid(images, 2, 4)
```

![img](https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/stable_diffusion_jax_how_to_cell_38_output_0.jpeg)

## Using different prompts

คุณไม่จำเป็นต้องใช้ prompt เดียวกันบนทุกอุปกรณ์. ตัวอย่างเช่น, เพื่อสร้าง 8 prompts ที่แตกต่างกัน:

```python
prompts = [
    "Labrador in the style of Hokusai",
    "Painting of a squirrel skating in New York",
    "HAL-9000 in the style of Van Gogh",
    "Times Square under water, with fish and a dolphin swimming around",
    "Ancient Roman fresco showing a man working on his laptop",
    "Close-up photograph of young black woman against urban background, high quality, bokeh",
    "Armchair in the shape of an avocado",
    "Clown astronaut in space, with Earth in the background",
]

prompt_ids = pipeline.prepare_inputs(prompts)
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, p_params, rng, jit=True).images
images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)

make_image_grid(images, 2, 4)
```

![img](https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/stable_diffusion_jax_how_to_cell_43_output_0.jpeg)


## How does parallelization work?

Flax pipeline in 🤗 Diffusers ทำการคอมไพล์โมเดลและทำงานแบบ parallel บนอุปกรณ์ที่มีอยู่ทั้งหมด. มาชมว่ากระบวนการนี้ทำงานอย่างไร.

การ parallelization ใน JAX สามารถทำได้หลายวิธี วิธีที่ง่ายที่สุดเกี่ยวข้องกับการใช้ [`jax.pmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) เพื่อให้ได้ single-program multiple-data (SPMD) parallelization หมายความว่าให้รันสำเนาของโค้ดเดียวกันหลายๆ ครั้ง, แต่ละครั้งมีข้อมูลนำเข้าที่แตกต่างกัน. มีวิธีที่ซับซ้อนมากมายและคุณสามารถดูไปที่เอกสารของ JAX [documentation](https://jax.readthedocs.io/en/latest/index.html) เพื่อศึกษาเรื่องนี้อย่างละเอียดหากคุณสนใจ!

`jax.pmap` ทำสองอย่าง:

1. คอมไพล์โค้ดซึ่งคล้ายกับ `jax.jit()`. นี้ไม่

เกิดขึ้นเมื่อคุณเรียกใช้ `pmap`, และเกิดเฉพาะครั้งแรกที่เรียกใช้ฟังก์ชันที่ได้รับการ `pmap`.
2. ทำให้โค้ดที่คอมไพล์เรียกใช้งานแบบ parallel บนอุปกรณ์ที่มีอยู่ทั้งหมด.

เพื่อตัวอย่าง, เรียกใช้ `pmap` บนเมธอด `_generate` ของ pipeline (นี้เป็นเมธอดส่วนตัวที่สร้างรูปภาพและอาจถูกเปลี่ยนชื่อหรือลบในการอัปเดตเวอร์ชันข้างหน้าของ 🤗 Diffusers):

```python
p_generate = pmap(pipeline._generate)
```

หลังจากเรียกใช้ `pmap`, ฟังก์ชันที่เตรียมไว้ `p_generate` จะ:

1. ทำสำเนาของฟังก์ชันใต้เพื่อ `pipeline._generate` บนแต่ละอุปกรณ์.
2. ส่งข้อมูลนำเข้าที่แตกต่างกันไปยังแต่ละอุปกรณ์ (นี่คือเหตุผลทำไมต้องเรียกใช้ฟังก์ชัน *shard*). ในที่นี้, `prompt_ids` มีรูปร่าง `(8, 1, 77, 768)` ดังนั้น array ถูกแบ่งเป็น 8 และแต่ละสำเนาของ `_generate` ได้รับข้อมูลนำเข้าที่มีรูปร่าง `(1, 77, 768)`.

สิ่งที่สำคัญที่สุดในที่นี้คือขนาด batch (1 ในตัวอย่างนี้), และมิติข้อมูลนำเข้าที่เหมาะสมสำหรับโค้ดของคุณ. คุณไม่ต้องเปลี่ยนอะไรเพิ่มเติมเพื่อให้โค้ดทำงานแบบ parallel.

การเรียก pipeline ครั้งแรกจะใช้เวลานานขึ้นเนื่องจากต้องคอมไพล์โค้ด, แต่การเรียกใช้ซ้ำ (แม้ด้วยข้อมูลนำเข้าที่แตกต่างกัน) เร็วขึ้นมาก. ฟังก์ชัน `block_until_ready` ถูกใช้เพื่อวัดเวลาการทำนายอย่างถูกต้องเนื่องจาก JAX ใช้การส่งหลังการโปรโมตและคืนควบคุมไปยังลูป Python โดยทันทีที่เป็นไปได้. คุณไม่ต้องใช้นั้นในโค้ดของคุณ; การบล็อกเกิดขึ้นโดยอัตโนมัติเมื่อคุณต้องการใช้ผลลัพธ์ของการคำนวณที่ยังไม่ได้เกิดขึ้น.

```py
%%time
images = p_generate(prompt_ids, p_params, rng)
images = images.block_until_ready()
"CPU times: user 1min 15s, sys: 18.2 s, total: 1min 34s"
"Wall time: 1min 15s"
```

ตรวจสอบขนาดภาพเพื่อดูว่าถูกต้อง:

```python
images.shape
"(8, 1, 512, 512, 3)"
```