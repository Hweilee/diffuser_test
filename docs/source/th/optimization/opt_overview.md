<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ภาพรวม

การสร้างผลลัพธ์ที่มีคุณภาพสูงเป็นการใช้ทรัพยากรทำงานเชิงคำนวณมากนัก โดยเฉพาะในทุกขั้นตอนที่คุณไปจากผลลัพธ์ที่มี noise ไปสู่ผลลัพธ์ที่น้อย noise ลง หนึ่งในวัตถุประสงค์ของ 🤗 Diffuser คือการทำให้เทคโนโลยีนี้สามารถเข้าถึงได้อย่างแพร่หลายสำหรับทุกคน ซึ่งรวมถึงการเปิดให้ใช้งานการอินเฟอเรนซ์ที่รวดเร็วบนฮาร์ดแวร์ที่เป็นของผู้บริโภคและพิเศษ

ในส่วนนี้จะกล่าวถึงเคล็มและเทคนิคต่างๆ - เช่น half-precision weights และ sliced attention - สำหรับการปรับปรุงความเร็วในการทำนายและลดการใช้หน่วยความจำ คุณยังจะเรียนรู้วิธีเพิ่มความเร็วให้โค้ด PyTorch ของคุณด้วย [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) หรือ [ONNX Runtime](https://onnxruntime.ai/docs/) และเปิดใช้งานการสนใจที่ประหยัดหน่วยความจำด้วย [xFormers](https://facebookresearch.github.io/xformers/) นอกจากนี้ยังมีคำแนะนำสำหรับการเรียกใช้การทำนายบนฮาร์ดแวร์ที่ระบุ เช่น Apple Silicon และ Intel หรือ Habana processors.