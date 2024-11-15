# Qwen2_VL 使用指南

<p align="left">
</p>

## 目录

- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载及转换](#jump2)
  - [权重下载](#jump2.1)
- [数据集准备及处理](#jump3)
  - [数据集下载](#jump3.1)
- [预训练](#jump4)
  - [准备工作](#jump4.1)
  - [配置参数](#jump4.2)
  - [启动预训练](#jump4.3)
- [推理](#jump5)
  - [准备工作](#jump5.1)
  - [配置参数](#jump5.2)
  - [启动推理](#jump5.3)

---
<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

|           软件            | [版本](https://www.hiascend.com/hardware/firmware-drivers/commercial?product=4&model=26) |
| :-----------------------: |:----------------------------------:|
|          硬件配置         |                Atlas 800T A2 <br> Atlas 900 A2 PoD                 |
|          Python           |                3.10                 |
|          Driver           |         AscendHDK 24.1.RC3          |
|         Firmware          |         AscendHDK 24.1.RC3          |
|           CANN            |             CANN 8.0.RC3             |
|           Torch           |            2.1.0            |
|         Torch_npu         |           release v6.0.RC3           |

<a id="jump1.1"></a>

#### 1. 仓库拉取

```shell
    git clone https://gitee.com/ascend/MindSpeed-MM.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../MindSpeed-MM/
    cd ..
    cd MindSpeed-MM
    mkdir logs
    mkdir data
    mkdir ckpt
```

<a id="jump1.2"></a>

#### 2. 环境搭建

torch npu 与 CANN包参考链接：[安装包参考链接](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)

```bash
    # python3.10
    conda create -n test python=3.10
    conda activate test

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.1.0-cp310-cp310m-manylinux2014_aarch64.whl 
    pip install torch_npu-2.1.0*-cp310-cp310m-linux_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    pip install apex-0.1_ascend*-cp310-cp310m-linux_aarch64.whl

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    # checkout commit from MindSpeed core_r0.6.0
    git checkout e9b34c97f62e628c8feb211c83ee4f493536ec11
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..
    # 安装其余依赖库
    pip install -e .
    #备注：当前需要修改下MindSpeed文件MindSpeed/mindspeed/core/transformer/transformer.py的368行，修改如下：
    #is_recompute_activation = should_recompute_activation(self.layer_number)
    #is_recompute_activation = should_recompute_activation(getattr(self, 'layer_number', None))
```

## 权重下载及转换

<a id="jump2.1"></a>

#### 1. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/tree/main)；

 将下载的模型权重保存到本地的`ckpt/hf_path/Qwen2-VL-7B-Instruct`目录下。
<a id="jump2.2"></a>

#### 2. 权重转换

MindSpeed-MM修改了部分原始网络的结构名称，使用examples/qwen2vl/qwen2vl_convert_to_mm_ckpt.py脚本对原始预训练权重进行转换。该脚本实现了从huggingface权重到MindSpeed-MM权重的转换以及PP（Pipeline Parallel）权重的切分 (目前只支持 7B 和 特定的切分方式)。

以Qwen2VL-7B为例
首先通过 [ModelLink](https://gitee.com/ascend/ModelLink) 的权重转换工具将 Qwen2VL-7B语言模型部分的权重转换到 megatron 支持的格式：

```
git clone https://gitee.com/ascend/ModelLink
cd ModelLink
```
将 modellink/tasks/checkpoint/models.py 的第393行
```
self.module = [AutoModelForCausalLM.from_pretrained(load_dir, device_map=device_map, trust_remote_code=trust_remote_code)]
```
改为：
```
from transformers import Qwen2VLForConditionalGeneration
self.module = [Qwen2VLForConditionalGeneration.from_pretrained(load_dir, device_map=device_map, trust_remote_code=trust_remote_code)]
```
创建权重转换脚本 modelconvert.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-tensor-parallel-size 1 \
    --add-qkv-bias \
    --load-dir hf_path/Qwen2-VL-7B-Instruct \
    --save-dir llm_path/Qwen2-VL-7B-Instruct \
    --tokenizer-model hf_path/Qwen2-VL-7B-Instruct/tokenizer.json \
    --model-type-hf llama2 \
    --params-dtype bf16
```

然后执行

```
bash modelconvert.sh
```

第二步，修改qwen2vl_convert_to_mm_ckpt.py中的load_dir、save_dir、pipeline_layer_index、num_layers、llm_path 如下：

```
hg_ckpt_dir = 'hf_path/Qwen2-VL-7B-Instruct' # huggingface权重目录
mm_save_dir = 'ckpt/Qwen2-VL-7B-Instruct'  # 转换后保存目录
pipeline_layer_index = None     # None表示不进行pp切分；若要进行pp切分，则需要传入一个列表，例如[0, 0, 10, 20]
num_layers=28                   # 模型结构层数
llm_path = 'llm_path/Qwen2-VL-7B-Instruct/inter_0000001/mp_rank/model_optim_rng.pt' # 第一步用 ModelLink 保存的模型路径
```
  
启动脚本

  ```
  # 根据实际情况修改 ascend-toolkit 路径
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python examples/qwen2vl/qwen2vl_convert_to_mm_ckpt.py
  ```

同步修改examples/qwen2vl/finetune_qwen2vl_7b.sh中的LOAD_PATH参数，该路径为转换后或者切分后的权重，注意与原始权重 hf_path/Qwen2-VL-7B-Instruct进行区分。

```
LOAD_PATH="ckpt/Qwen2-VL-7B-Instruct"
```

<a id="jump3"></a>

## 数据集准备及处理

<a id="jump3.1"></a>

#### 1. 数据集下载(以coco2017数据集为例)

(1)用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中

(2)获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下;

(3)在./data路径下新建文件mllm_format_llava_instruct_data.json，运行数据转换脚本python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py;

   ```
   $playground
   ├── data
       ├── COCO2017
           ├── train2017
           
       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

---

<a id="jump4"></a>

## 微调

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 配置参数

【数据目录配置】

根据实际情况修改`data.json`中的数据集路径，包括`model_name_or_path`、`dataset_dir`、`dataset`等字段。

以Qwen2VL-7B为例，`data.json`进行以下修改，注意`model_name_or_path`的权重路径为转换前的权重路径。

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",
        "preprocess_parameters": {
            "model_name_or_path": "./ckpt/hf_path/Qwen2-VL-7B-Instruct",
            ...
        },
        "basic_parameters": {
            ...
            "dataset_dir": "./data",
            "dataset": "./data/mllm_format_llava_instruct_data.json",
            "cache_dir": "./data/cache_dir",
            ...
        },
        ...
    },
    ...
    }
}
```

【模型保存加载配置】

根据实际情况配置`examples/qwen2vl/finetune_qwen2vl_7b.sh`的参数，包括加载、保存路径以及保存间隔`--save-interval`（注意：分布式优化器保存文件较大耗时较长，请谨慎设置保存间隔）

```shell
...
# 加载路径
LOAD_PATH="ckpt/Qwen2-VL-7B-Instruct"
# 保存路径
SAVE_PATH="save_dir"
...
GPT_ARGS="
    ...
    --no-load-optim \  # 不加载优化器状态，若需加载请移除
    --no-load-rng \  # 不加载随机数状态，若需加载请移除
    --no-save-optim \  # 不保存优化器状态，若需保存请移除
    --no-save-rng \  # 不保存随机数状态，若需保存请移除
    ...
"
...
OUTPUT_ARGS="
    --log-interval 1 \  # 日志间隔
    --save-interval 5000 \  # 保存间隔
    ...
"
```

若需要加载指定迭代次数的权重、优化器等状态，需将加载路径`LOAD_PATH`设置为保存文件夹路径`LOAD_PATH="save_dir"`，并修改`latest_checkpointed_iteration.txt`文件内容为指定迭代次数
(此功能coming soon)

```
$save_dir
   ├── latest_checkpointed_iteration.txt
   ├── ...
```

【单机运行配置】

配置`examples/qwen2vl/finetune_qwen2vl_7b.sh`参数如下

```shell
    # 根据实际情况修改 ascend-toolkit 路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 
    GPUS_PER_NODE=8
    MASTER_ADDR=locahost
    MASTER_PORT=29501
    NNODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

<a id="jump4.3"></a>

#### 3. 启动微调

以Qwen2VL-7B为例，启动微调训练任务。

```shell
    bash examples/qwen2vl/finetune_qwen2vl_7b.sh
```

<a id="jump5"></a>

## 推理

Coming Soon...
