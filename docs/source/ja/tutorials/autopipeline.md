<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# AutoPipeline

Diffusersは様々なタスクをこなすことができ、テキストから画像、画像から画像、画像の修復など、複数のタスクに対して同じように事前学習された重みを再利用することができます。しかし、ライブラリや拡散モデルに慣れていない場合、どのタスクにどのパイプラインを使えばいいのかがわかりにくいかもしれません。例えば、 [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) チェックポイントをテキストから画像に変換するために使用している場合、それぞれ[`StableDiffusionImg2ImgPipeline`]クラスと[`StableDiffusionInpaintPipeline`]クラスでチェックポイントをロードすることで、画像から画像や画像の修復にも使えることを知らない可能性もあります。

`AutoPipeline` クラスは、🤗 Diffusers の様々なパイプラインをよりシンプルするために設計されています。この汎用的でタスク重視のパイプラインによってタスクそのものに集中することができます。`AutoPipeline` は、使用するべき正しいパイプラインクラスを自動的に検出するため、特定のパイプラインクラス名を知らなくても、タスクのチェックポイントを簡単にロードできます。

<Tip>

どのタスクがサポートされているかは、[AutoPipeline](../api/pipelines/auto_pipeline) のリファレンスをご覧ください。現在、text-to-image、image-to-image、inpaintingをサポートしています。

</Tip>

このチュートリアルでは、`AutoPipeline` を使用して、事前に学習された重みが与えられたときに、特定のタスクを読み込むためのパイプラインクラスを自動的に推測する方法を示します。

## タスクに合わせてオートパイプラインを選択する
まずはチェックポイントを選ぶことから始めましょう。例えば、 [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) チェックポイントでテキストから画像への変換したいなら、[`AutoPipelineForText2Image`]を使います:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")
prompt = "peasant and dragon combat, wood cutting style, viking era, bevel with rune"

image = pipeline(prompt, num_inference_steps=25).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-text2img.png" alt="generated image of peasant fighting dragon in wood cutting style"/>
</div>
[`AutoPipelineForText2Image`]を具体的に見ていきましょう:

1. [`model_index.json`](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/model_index.json) ファイルから `"stable-diffusion"` クラスを自動的に検出します。
2. `"stable-diffusion"` のクラス名に基づいて、テキストから画像へ変換する [`StableDiffusionPipeline`] を読み込みます。

同様に、画像から画像へ変換する場合、[`AutoPipelineForImage2Image`] は `model_index.json` ファイルから `"stable-diffusion"` チェックポイントを検出し、対応する [`StableDiffusionImg2ImgPipeline`] を読み込みます。また、入力画像にノイズの量やバリエーションの追加を決めるための強さなど、パイプラインクラスに固有の追加引数を渡すこともできます:

```py
from diffusers import AutoPipelineForImage2Image
import torch
import requests
from PIL import Image
from io import BytesIO

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")
prompt = "a portrait of a dog wearing a pearl earring"

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/800px-1665_Girl_with_a_Pearl_Earring.jpg"

response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image.thumbnail((768, 768))

image = pipeline(prompt, image, num_inference_steps=200, strength=0.75, guidance_scale=10.5).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-img2img.png" alt="generated image of a vermeer portrait of a dog wearing a pearl earring"/>
</div>

また、画像の修復を行いたい場合は、 [`AutoPipelineForInpainting`] が、同様にベースとなる[`StableDiffusionInpaintPipeline`]クラスを読み込みます：

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

prompt = "A majestic tiger sitting on a bench"
image = pipeline(prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-inpaint.png" alt="generated image of a tiger sitting on a bench"/>
</div>

サポートされていないチェックポイントを読み込もうとすると、エラーになります:

```py
from diffusers import AutoPipelineForImage2Image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "openai/shap-e-img2img", torch_dtype=torch.float16, use_safetensors=True
)
"ValueError: AutoPipeline can't find a pipeline linked to ShapEImg2ImgPipeline for None"
```

## Use multiple pipelines

For some workflows or if you're loading many pipelines, it is more memory-efficient to reuse the same components from a checkpoint instead of reloading them which would unnecessarily consume additional memory. For example, if you're using a checkpoint for text-to-image and you want to use it again for image-to-image, use the [`~AutoPipelineForImage2Image.from_pipe`] method. This method creates a new pipeline from the components of a previously loaded pipeline at no additional memory cost.

The [`~AutoPipelineForImage2Image.from_pipe`] method detects the original pipeline class and maps it to the new pipeline class corresponding to the task you want to do. For example, if you load a `"stable-diffusion"` class pipeline for text-to-image:

```py
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch

pipeline_text2img = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
print(type(pipeline_text2img))
"<class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'>"
```

Then [`~AutoPipelineForImage2Image.from_pipe`] maps the original `"stable-diffusion"` pipeline class to [`StableDiffusionImg2ImgPipeline`]:

```py
pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img)
print(type(pipeline_img2img))
"<class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline'>"
```

If you passed an optional argument - like disabling the safety checker - to the original pipeline, this argument is also passed on to the new pipeline:

```py
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch

pipeline_text2img = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
    requires_safety_checker=False,
).to("cuda")

pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img)
print(pipeline_img2img.config.requires_safety_checker)
"False"
```

You can overwrite any of the arguments and even configuration from the original pipeline if you want to change the behavior of the new pipeline. For example, to turn the safety checker back on and add the `strength` argument:

```py
pipeline_img2img = AutoPipelineForImage2Image.from_pipe(pipeline_text2img, requires_safety_checker=True, strength=0.3)
print(pipeline_img2img.config.requires_safety_checker)
"True"
```
