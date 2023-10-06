<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# การติดตั้ง

ติดตั้ง 🤗 Diffusers สำหรับไลบรารี deep learning ที่คุณกำลังทำงานอยู่

🤗 Diffusers ผ่านการทดสอบบน Python 3.8+, PyTorch 1.7.0+ และ Flax โปรดทำตามคำแนะนำการติดตั้งด้านล่างสำหรับไลบรารี deep learning ที่คุณกำลังใช้:

- [PyTorch](https://pytorch.org/get-started/locally/) คำแนะนำการติดตั้ง
- [Flax](https://flax.readthedocs.io/en/latest/) คำแนะนำการติดตั้ง

## ติดตั้งด้วย pip

คุณควรติดตั้ง 🤗 Diffusers ใน [virtual environment](https://docs.python.org/3/library/venv.html).
หากคุณไม่เคยใช้งาน Python virtual environments ดูที่ [คู่มือนี้](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
การใช้งาน virtual environment ทำให้การจัดการโปรเจกต์ต่าง ๆ และป้องกันปัญหาความไม่เข้ากันของ dependencies ได้ง่ายขึ้น

เริ่มต้นด้วยการสร้าง virtual environment ในไดเรกทอรีของโปรเจกต์ของคุณ:

```bash
python -m venv .env
```

เปิดใช้งาน virtual environment:

```bash
source .env/bin/activate
```

🤗 Diffusers ยังพึ่งพาที่ไลบรารี 🤗 Transformers และคุณสามารถติดตั้งทั้งคู่ด้วยคำสั่งต่อไปนี้:

<frameworkcontent>
<pt>
```bash
pip install diffusers["torch"] transformers
```
</pt>
<jax>
```bash
pip install diffusers["flax"] transformers
```
</jax>
</frameworkcontent>

## ติดตั้งจากแหล่งที่มา

ก่อนที่จะติดตั้ง 🤗 Diffusers จากแหล่งที่มา โปรดตรวจสอบให้แน่ใจว่าคุณได้ติดตั้ง `torch` และ 🤗 Accelerate ไว้แล้ว

สำหรับการติดตั้ง `torch` โปรดดูที่คู่มือการ [ติดตั้ง](https://pytorch.org/get-started/locally/#start-locally) ของ `torch`.

เพื่อทำการติดตั้ง 🤗 Accelerate:

```bash
pip install accelerate
```

ติดตั้ง 🤗 Diffusers จากแหล่งที่มาด้วยคำสั่งต่อไปนี้:

```bash
pip install git+https://github.com/huggingface/diffusers
```

คำสั่งนี้จะทำการติดตั้งเวอร์ชัน `main` ล่าสุดแทนที่จะเป็นเวอร์ชัน `stable` ล่าสุด
เวอร์ชัน `main` มีประโยชน์ในการอัปเดตกับข้อมูลที่พัฒนาล่าสุด
ตัวอย่างเช่น หากมีการแก้บั๊กตั้งแต่การปล่อยเวอร์ชันออกมาครั้งล่าสุดแต่ยังไม่ได้มีการปล่อยเวอร์ชันใหม่
อย่างไรก็ตาม นี้หมายความว่าเวอร์ชัน `main` อาจจะไม่เสถียรเสมอไป
เรามุ่งมั่นที่จะให้เวอร์ชัน `main` ใช้งานได้เสมอ และปัญหาส่วนใหญ่มักจะได้รับการแก้ไขภายในไม่กี่ชั่วโมงหรือหนึ่งวัน
หากคุณพบปัญหา โปรดเปิด [Issue](https://github.com/huggingface/diffusers/issues/new/choose) เพื่อให้เราแก้ไขโดยรวดเร็ว!

## การติดตั้งเพื่อแก้ไข

คุณจะต้องทำการติดตั้งแบบที่สามารถแก้ไขได้หากคุณต้องการ:

* ใช้เวอร์ชัน `main` ของโค้ดต้นฉบับ
* มีส่วนที่จะมีการเปลี่ยนแปลงในโค้ดของ 🤗 Diffusers และต้องทดสอบการเปลี่ยนแปลงในโค้ด

โคลน repository และติดตั้ง 🤗 Diffusers ด้วยคำสั่งต่อไปนี้:

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
```

<frameworkcontent>
<pt>
```bash
pip install -e ".[torch]"
```
</pt>
<jax>
```bash
pip install -e ".[flax]"
```
</jax>
</frameworkcontent>

คำสั่งเหล่านี้จะเชื่อมโยงโฟลเดอร์ที่คุณโคลน repository ไปยังเส้นทางไลบรารีของ Python ของคุณ
ตอนนี้ Python จะดูภายในโฟลเดอร์ที่คุณโคลนมาในเพิ่มเติมจากเส้นทางไลบรารีปกติ
ตัวอย่างเช่น หาก package ของ Python ของคุณติดตั้งโดยปกติที่ `~/anaconda3/envs/main/lib/python3.8/site-packages/`, Python จะค้นหาไฟล์ในโฟลเดอร์ `~/diffusers/` ที่คุณโคลนมา

<Tip warning={true}>

คุณต้องเก็บโฟลเดอร์ `diffusers` ถ้าคุณต้องการใช้ไลบรารีต่อไป

</Tip>

ตอนนี้คุณสามารถอัปเดตโคลนของคุณไปยังเวอร์ชันล่าสุดของ 🤗 Diffusers ได้โดยใช้คำสั่งต่อไปนี้:

```bash
cd ~/diffusers/
git pull
```

สภาพแวดล้อม Python ของคุณจะพบเวอร์ชัน `main` ของ 🤗 Diffusers ในการเรียกใช้ครั้งถัดไป

## ประกาศเกี่ยวกับการเก็บข้อมูล

ไลบรารีของเราเก็บข้อมูลโทรเลขระหว่างการร้องขอ `from_pretrained()`
ข้อมูลเหล่านี้รวมถึงเวอร์ชันของ Diffusers และ PyTorch/Flax, ร้องขอโมเดลหรือคลาสไพป์ไลน์,
และเส้นทางไปยังจุดตรวจสอบก่อนการฝึกอบรมหากมีการโฮสต์อยู่บน Hub
ข้อมูลการใช้นี้ช่วยให้เราแก้ปัญหาและกำหนดลำดับความสำคัญของคุณลักษณะใหม่
ข้อมูลโทรเลขจะถูกส่งเฉพาะเมื่อโหลดโมเดลและไพป์ไลน์จาก HuggingFace Hub
และไม่มีการเก็บข้อมูลเมื่อใช้ท้องถิ่น

เรารู้ว่าไม่ทุกคนต้องการแชร์ข้อมูลเพิ่มเติม และเรารีสเปคความเป็นส่วนตัวของคุณ
ดังนั้นคุณสามารถปิดการเก็บข้อมูลโทรเลขได้โดยการตั้งค่าตัวแปรสภาพแวดล้อม `DISABLE_TELEMETRY` จาก terminal ของคุณ:

บน Linux/MacOS:
```bash
export DISABLE_TELEMETRY=YES
```

บน Windows:
```bash
set DISABLE_TELEMETRY=YES
```
