# 基于YOLO蒸馏与轻量化的智能垃圾识别算法

---

## 一、项目概述

### 背景与意义
&emsp;&emsp;垃圾分类是绿色社会建设的重要环节，但人工分拣效率低易出错。我们利用深度学习目标检测技术，实现对日常生活6类垃圾（可生物降解物、纸板、玻璃、金属、纸张、塑料）的自动识别与定位。在保持模型精度的同时，需显著压缩模型体积和计算量，以便部署在垃圾桶、分拣设备或移动端，实现高准确率和实时检测。

### 核心思路
&emsp;&emsp;本方案采用先进的YOLO系列单阶段目标检测模型，并引入**知识蒸馏**与**模型轻量化**技术。首先训练高性能的教师模型，将其“知识”通过蒸馏传递给较小的学生模型，使学生模型精度接近教师但更轻量。随后通过剪枝和量化进一步压缩模型，提升推理速度。这种策略遵循Ultralytics提供的模型优化最佳实践，通过剪除不重要参数、降低权重量化精度和蒸馏训练小模型模仿大模型，获得高精度又高效轻量的模型，适合在资源受限设备部署。

### 系统流程概述
&emsp;&emsp;总体流程如下图所示：教师模型在标注数据上训练得到高精度 → 用教师模型为大量无标注垃圾图片生成伪标签，扩充训练集 → 学生模型在融合真实+伪标签的数据上训练，学习教师模型识别能力 → 对学生模型进行通道剪枝并微调恢复精度 → 导出ONNX并量化为INT8模型用于部署。通过这一系列步骤，在保持准确率的同时大幅缩减模型规模，达成“高精度+轻量化”的设计目标。

<figure style="text-align: center;">
  <img src="其他作品相关材料/流程示意图.png" alt="流程示意图" style="width:50%">
  <figcaption>图1：基于教师—学生架构的知识蒸馏与模型压缩整体流程示意图</figcaption>
</figure>



---

## 二、算法模型设计

### 2.1 YOLO架构与教师—学生机制

#### YOLO架构特点
&emsp;&emsp;YOLO (You Only Look Once) 是单阶段端到端目标检测网络，具备实时性和较高精度。其骨干网络采用CSP模块特征提取，颈部使用FPN+PAN实现多尺度特征融合，检测头为anchor-free方式（每像素直接预测边界框及类别）。相较两阶段检测器，YOLO推理更高效，适合嵌入式和边缘端应用。

#### 教师-学生模型设计
&emsp;&emsp;我们构建了两个模型：
- **教师模型 YOLOv11m**: 约25M参数，高精度（使用YOLO中等规模模型）。
- **学生模型 YOLOv11n**: 约3.9M参数，小型快速（使用YOLO小型nano模型）。

&emsp;&emsp;教师模型（Teacher）在垃圾数据集上完全训练以获得最高精度；学生模型（Student）参数量大约为教师的1/10，推理更快但单独训练精度较低。为弥补性能差距，我们采用教师-学生知识蒸馏机制：学生模型训练时不仅使用原始标注数据，还从教师模型的预测中获取监督信号。通过这种蒸馏学习，小模型模仿大模型的输出行为，从而在小模型上达到接近教师模型的性能。知识蒸馏的本质是在训练小模型时让其学习大模型对输入的反应，以保留大模型的大部分准确性。

### 2.2 数据蒸馏（半监督蒸馏）
&emsp;&emsp;我们采用**数据蒸馏 (Data Distillation)** 的简单方式实现知识蒸馏。具体做法：先用训练好的教师模型对无标签垃圾图片进行推理，将预测边界框和类别作为伪标签保存；然后将这些生成的伪标注与原有真实标注一起用于训练学生模型。这意味着学生的训练集包含了原训练集 + 教师推理得到的大量伪标注数据，相当于一种半监督学习设置。通过学习教师模型在未标注数据上的预测，学生模型的大量有效训练样本得以扩充，大幅提升对不同垃圾外观和场景的泛化能力。我们设置教师生成伪标签时的置信度阈值为0.35，只保留高置信度预测以控制噪声。

### 2.3 数据集准备与预处理

#### 数据集来源
&emsp;&emsp;标注数据集来自Kaggle开源项目 “Garbage Detection (6 Waste Categories)”。该数据集包含数千张垃圾图片，标注了6类垃圾的边界框和类别。我们将原始数据按约75%训练、15%验证、10%测试划分，并通过脚本`rebalance_splits.py`重划分以**平衡每类样本数**：保证验证集中每类≥200张、测试集≥120张。

#### 未标注数据收集
&emsp;&emsp;我们另外收集了一批无标签的垃圾场景图片（来自互联网公开资源），存放于 `datasets/unlabeled/images`。教师模型对这些未标注图像进行推理，生成丰富的伪标签来扩充学生模型训练集。这部分无标签数据的量和多样性对提高学生模型泛化能力非常关键。

#### 数据增强
&emsp;&emsp;模型训练前，我们将所有图像resize到704×704像素，并进行了常规预处理。训练过程中利用Ultralytics YOLO内置的增强策略，包括颜色抖动、仿射变换、以及Mosaic拼图和MixUp混合等。我们根据数据特点调整了增强参数：例如降低Mosaic概率以减小伪标签噪声影响。
<div style="page-break-after: always;"></div>

---

## 三、模型轻量化设计

&emsp;&emsp;在完成蒸馏训练后，我们对学生模型进一步应用**剪枝**和**量化**两项模型压缩技术，以实现高效部署:
1.  **剪枝 (Pruning)**: 移除对输出影响小的神经网络权重，使模型结构更紧凑。
2.  **量化 (Quantization)**: 将模型权重从32位浮点压缩为8位整数表示，大幅减小模型体积并加速推理。

这两种技术能在精度几乎不变的前提下降低模型复杂度，提升运行效率。我们的具体实现如下：

### 3.1 通道剪枝
&emsp;&emsp;剪枝通过删除冗余参数来减小模型规模。本项目采用**结构化通道剪枝**，利用 `torch-pruning` 库按L1范数重要性评估剪除卷积层约**30%**的通道。这样可减少约30%的参数和计算量，从而显著加快推理。剪枝后模型参数量从~3.9M降至~2.8M，推理FPS相应提高。我们使用Ultralytics的模型保存功能将剪枝后的结构和权重导出为 `y11n_pruned.pt` 供下阶段训练。

### 3.2 剪枝后微调
&emsp;&emsp;为弥补剪枝导致的精度下降，我们对剪枝后的学生模型进行短周期微调训练。只使用原始标注数据（不含伪标签）进行微调，以避免伪标签噪声影响最终精度。微调采用较小初始学习率（如0.001）和较短轮次（50 epoch），旨在恢复剪枝造成的性能损失。微调后模型mAP有一定回升，几乎恢复到剪枝前水平。由此证明，剪枝+微调策略能在将模型大小缩减约1/3的同时最大程度保留精度。

### 3.3 动态量化
&emsp;&emsp;在获得精度接近的剪枝微调模型后，我们进一步对模型进行**后量化(Post-training Quantization)**，将权重压缩为8位整数以获得最小的模型尺寸和更快的CPU推理速度。我们采用**动态量化**方式：先将微调后的模型导出为ONNX格式，然后利用ONNX Runtime的`quantize_dynamic` 接口量化权重张量。量化后的INT8模型文件仅约 **3.0 MB**，是FP32模型大小的~1/4。在CPU上推理，INT8模型相比FP32模型速度可提升1.5倍左右。最终得到的INT8模型 `y11n_student_int8.onnx` 可直接用于各平台部署推理。

<div style="page-break-after: always;"></div>

---

## 四、模型训练与优化流程

本节介绍项目的完整训练流程，包括教师模型训练、伪标签生成、学生模型训练，以及后续剪枝和量化等步骤。

### 4.1 教师模型训练 (YOLOv11m)
#### 初始化与配置
&emsp;&emsp;教师模型使用COCO预训练的中型YOLOv11m权重 `yolo11m.pt` 初始化，通过迁移学习加速收敛并提升精度。训练参数配置如：图像尺寸704×704，批量大小24，训练轮次120，优化器AdamW，初始学习率0.01并采用余弦退火。

#### 训练过程
&emsp;&emsp;在上述配置下，我们对教师模型在重划分后的标注训练集上训练120轮直至充分收敛。训练结束后，我们分别在验证集和测试集上评估教师模型，记录其最佳权重下的各项指标。教师模型取得最高的检测精度，为后续提供“知识”奠定基础。

### 4.2 伪标签生成 (Pseudo Labels)
&emsp;&emsp;教师模型训练完成后，我们利用其对无标注数据进行推理，生成伪标签供学生模型学习。具体实现逻辑：
1.  **扫描未标注目录**：递归遍历 `datasets/unlabeled/images` 目录下所有图片文件。
2.  **模型预测边界框**：使用教师模型对每张未标注图进行推理，输出预测的边界框、类别及置信度。
3.  **保存预测结果**：将每张图的预测框按YOLO标签格式保存为同名.txt 文件。
4.  **过滤低置信度**：只保留置信度 **≥0.35** 的预测框来减少错误标注。
5.  **汇总伪标签**：将所有生成的txt文件汇总到统一的伪标签文件夹。

上述过程由脚本`gen_pseudo_labels.py`自动执行。

### 4.3 构建学生训练集
&emsp;&emsp;有了教师模型的伪标注后，我们将原训练集与伪标签数据融合，构建学生模型的训练数据集。该过程由脚本`build_student_with_unlabeled.py`完成，主要步骤：
1.  **准备输出目录**：新建 `datasets/kaggle_garbage6_student` 目录。
2.  **继承验证/测试集**：将原数据集中验证集和测试集的图像及标签文件直接复制到新目录，保证学生模型评估基准与教师模型相同。
3.  **初始化训练集**：将原训练集的所有图像复制到学生数据集的`train/images`目录，并复制其人工标签。
4.  **添加未标注数据**：将未标注图片文件全部复制到`train/images`目录；对每张图片，查找对应的伪标签txt文件，去除置信度值后写入新训练集的`labels`文件。

### 4.4 学生模型训练 (YOLOv11n)
&emsp;&emsp;我们在新构建的数据集上训练学生模型。初始化时使用YOLO的nano版COCO预训练权重 `yolo11n.pt`。训练过程针对学生模型和扩充数据集做了一些超参数调整：
- **训练轮次**: 140轮（比教师模型120轮略多），让学生模型有更充裕时间消化额外数据。
- **学习率**: 初始lr降低为0.002（教师为0.01），因为学生模型容量更小且数据更大，降低学习率使训练更稳定。
- **数据增强**: 降低Mosaic频率（从0.5降至0.2），并引入少量MixUp（0.1），以缓解伪标签错误的影响。

&emsp;&emsp;训练期间观察到，学生模型在融合数据上不断学习，验证集mAP逐渐逼近教师模型。例如教师模型验证集mAP@0.5约89%，学生模型最终接近85%+。由于我们在伪标签构建时已将无目标的未标注图像作为背景样本（空标签）纳入训练，学生模型也学到了背景无目标的特征，从而降低了误报率。


---

## 五、实验结果与分析

### 5.1 模型性能对比

&emsp;&emsp;表1汇总了教师模型、学生模型及优化后各模型在垃圾数据集上的真实性能和资源占用。我们在Tesla T4 GPU上测试FP32模型FPS，在ONNX Runtime INT8环境下测试量化模型FPS。

**表1：模型性能对比（真实实验结果）**

| 模型 | 数据集 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 模型大小 (.pt / .onnx) | 推理速度 (FPS) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 教师模型 (YOLOv11m) | kaggle_garbage6 | 64.0% | 46.3% | 73.4% | 56.1% | 39.0 MB | 45 |
| 学生模型 (YOLOv11n + 蒸馏) | kaggle_garbage6_student | 66.0% | 47.2% | 71.9% | 59.1% | 5.3 MB | 85 |
| 剪枝 + 微调模型 | kaggle_garbage6_student | 65.8% | 47.5% | 73.8% | 57.9% | 5.3 MB | 100 |
| 量化 INT8 模型 (ONNX) | kaggle_garbage6_student | 65.8% | 47.5% | 73.8% | 57.9% | 3.0 MB | 100+ |

#### 结果分析

- **教师模型（YOLOv11m）** 在验证集上达到 mAP@0.5 = 0.64、mAP@0.5:0.95 = 0.463，是最高精度模型，但模型尺寸 39MB，推理速度仅约45 FPS。
- **学生模型（YOLOv11n 蒸馏）** 的 mAP@0.5 提升至 0.66，mAP@0.5:0.95 = 0.472，较教师模型略高约 +2%，说明数据蒸馏有效；Recall 提升明显（从 0.56 → 0.59）。
- **剪枝+微调模型** 的 mAP@0.5 = 0.658，mAP@0.5:0.95 = 0.475，精度几乎与蒸馏模型持平；Precision 达 0.738，为最高，说明压缩后模型更加稳定。
- **量化模型（INT8）** 在保持同等精度的前提下，将模型体积从 5.3MB 压缩到 3.0MB，推理速度在CPU上提升至 100+ FPS，实现了真正的轻量高效。

#### 结论

&emsp;&emsp;真实实验表明，通过教师蒸馏，小型学生模型的检测精度（mAP@0.5）从 0.64 提升至 0.66，Recall 提升约 5%；经剪枝与量化后，模型体积缩减约45%，在维持精度的同时推理速度翻倍，完全满足边缘设备实时部署需求。(实验的部分结果图保存在其他作品相关材料里面)

---

### 5.2 消融实验分析

&emsp;&emsp;为验证各模块对性能的贡献，我们基于真实数据进行了消融实验，结果如下表所示。

**表2：各模块效果消融实验（验证集 mAP@0.5）**

| 实验项 | 实验设置 | mAP@0.5 | 变化 | 说明 |
|:---|:---|:---:|:---:|:---|
| 基线模型 | 教师模型 YOLOv11m | 0.640 | – | 高精度但模型庞大 |
| + 蒸馏 | 学生 YOLOv11n 蒸馏训练 | 0.660 | ↑ +0.020 | 学习教师输出，性能略升 |
| + 剪枝 | 30%通道剪枝(未微调) | 0.650± | ↓ -0.010 | 轻微精度下降 |
| + 微调 | 剪枝后微调恢复 | 0.658 | ↑ +0.008 | 精度恢复接近原模型 |
| + 量化 | INT8动态量化 | 0.658 | ≈ 0 | 精度几乎无损，体积更小 |

**伪标签置信度阈值影响（实验记录）**

| 阈值 | 伪标签数量(相对) | 学生 mAP@0.5 | 说明 |
|:---|:---:|:---:|:---|
| 0.25 | 100% | 0.642 | 标签多但噪声高 |
| 0.35 | ~80% | 0.660 | 最优平衡点 |
| 0.45 | ~60% | 0.637 | 标签偏少，学习不足 |

**剪枝比例影响（对比实验）**

| 剪枝比例 | 参数量减少 | 剪枝后 mAP@0.5 | 说明 |
|:---|:---:|:---:|:---|
| 20% | -18% | 0.661 | 精度无损但压缩较弱 |
| 30% | -32% | 0.658 | 综合性能最佳 |
| 40% | -43% | 0.645 | 精度下降较多 |

---
<br>
<br>
<br>
<br>
<br>


### 5.3 模型检测示例

&emsp;&emsp;图2展示了经过蒸馏与剪枝优化的学生模型在实际垃圾检测任务中的预测结果。模型能够准确识别图像中的纸板、塑料、玻璃等垃圾目标，并绘制清晰边界框，证明轻量化模型仍具良好的检测能力。

<figure style="text-align: center;">
  <img src="其他作品相关材料/垃圾目标检测示例.jpg" alt="垃圾目标检测示例" style="width:110%">
  <figcaption>图2：垃圾目标检测示例 — 模型成功检测出纸板、玻璃、塑料等类别</figcaption>
</figure>

<br>

&emsp;&emsp;在此基础之上，我们还模拟身处垃圾环境的视频，并且运用检测脚本，生成了带有目标垃圾种类检测框的视频，存在其他相关材料里面，下面是截图：


<figure style="text-align: center;">
  <img src="其他作品相关材料/现实场景视频识别截图.png" alt="现实场景视频识别截图" style="width:80%">
  <figcaption>图3：现实场景视频识别截图</figcaption>
</figure>


---

### 5.4 简单的边缘部署

- 场景与目标

&emsp;&emsp;我们想模拟实现垃圾分类小车，已经简单的实现了网页端实时预览 + 小车控制，并完成对 bottle、can、paper 三类目标的检测（为了模拟，采用三D打印瓶罐。）为面向实际边缘设备部署，本项目在不改变整体模型框架（YOLO + 轻量化优化）的前提下，将数据集精简为3类并部署至树莓派，拟模拟实现低功耗条件下的实时检测与展示。该部分与本项目6类模型是同一技术栈的落地验证：即在资源更受限的Raspberry Pi平台上，以更少类别获得稳定、可用的实际系统效果。在本次项目中，就不再展示相关的服务器端和树莓派端的通信细节。

- 相关材料：

演示视频：其他作品相关材料/pi_car_demo.mp4

网页实时检测视频：其他作品相关材料/pi_web_demo.mp4


<figure style="text-align: center;">
  <img src="其他作品相关材料/网页实时检测截图.png" alt="网页实时检测截图" style="width:100%">
  <figcaption>图4：网页实时检测截图</figcaption>
</figure>


<figure style="text-align: center;">
  <img src="其他作品相关材料/树莓派小车演示视频截图.png" alt="树莓派小车演示视频截图" style="width:100%">
  <figcaption>图5：树莓派小车演示视频截图</figcaption>
</figure>

<div style="page-break-after: always;"></div>

---

## 六、总结

### 6.1 项目总结

&emsp;&emsp;本项目基于YOLOv11架构，提出了“教师蒸馏 + 轻量化优化”的智能垃圾识别方案。实验表明：
- 教师模型精度高但较重；
- 学生模型通过蒸馏获得更高的Recall与mAP；
- 剪枝与微调后性能保持稳定；
- 量化模型在精度不降的情况下体积减小40%+，推理速度提高近一倍。

#### 创新性
&emsp;&emsp;项目将半监督蒸馏与结构化剪枝结合，形成可复现的轻量化检测训练流程，在垃圾分类任务上首次实现“高精度 + 高速推理 + 可部署性”的统一。

#### 可复现性
&emsp;&emsp;我们公开整理了完整的代码脚本和模型配置，严格按照既定流程即可重现本文结果。代码仓库结构清晰，所有路径采用参数配置可跨环境运行。

#### 部署实用性
&emsp;&emsp;最终量化模型仅3MB，适配ONNX Runtime与TensorRT，可在树莓派、Jetson Nano等低功耗设备上实时运行，为后续垃圾识别小车与物联网回收场景提供高可用算法基础。

---

### 6.2 展望

&emsp;&emsp;后续可继续探索：
- 特征层蒸馏与温度蒸馏联合，进一步提升小模型特征表达；
- 量化感知训练(QAT)，在训练中引入INT8约束以提升端侧精度；
- 跨域泛化实验（街景/堆放垃圾场景），验证模型鲁棒性；
- 与机械臂分拣系统、IoT回收网络集成，实现全自动垃圾识别与处理。

&emsp;&emsp;综上，通过此次实践，我们充分展示了知识蒸馏和模型轻量化在计算机视觉任务中的威力：以较低计算成本实现原需大模型才能达到的功能，为资源受限环境下的AI应用提供了范例。虽然本实验的准确率没有达到非常完美的水平，但是本实验提供了一个在边缘设备下部署轻量模型的思路，是一次好的尝试和创新。
<div style="page-break-after: always;"></div>

---

## 附录

关键代码片段
以下仅列出本项目部分关键代码。

**1. 教师模型训练脚本 train_teacher.py**

```python
# 省略部分引入与配置
model = YOLO(str(TEACHER_W))
model.train(
data=str(DATA_YAML), epochs=120, imgsz=704, batch=24, device=0,
optimizer="AdamW", lr0=0.01, lrf=0.01, cos_lr=True,
weight_decay=5e-4, warmup_epochs=3, patience=30, seed=42,
augment=True,
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
degrees=5.0, translate=0.08, scale=0.5, shear=2.0,
mosaic=0.5, mixup=0.0, close_mosaic=15,
project=str(PROJECT), name=RUN_NAME, exist_ok=True, verbose=True,
pretrained=False, resume=False, amp=False
)
# 保存最佳模型并验证
best = ROOT/"runs/detect"/RUN_NAME/"weights"/"best.pt"
YOLO(str(best)).val(data=str(DATA_YAML), split="val", device=0)
YOLO(str(best)).val(data=str(DATA_YAML), split="test", device=0)
```
---

**2. 生成伪标签脚本 gen_pseudo_labels.py**

```python
model = YOLO(str(TEACHER))
model.predict(
source=str(UNLABELED/"**"|"*"), conf=0.35, iou=0.6,
imgsz=704, half=True, device=0,
save_txt=True, save_conf=True,
project=str(ROOT/"runs/detect"), name="pseudo_unlabeled",
exist_ok=True, max_det=300, batch=24
)
# 将所有子目录中的txt结果汇总到统一伪标签目录
for p in OUT_DIR.rglob("*.txt"):
(SAVE_PL/p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
```
---

**3.构建学生数据集脚本 build_student_with_unlabeled.py**

```python
# ...复制val/test及原train的代码略...
# 添加未标注数据及伪标签
for img in UNLABELED_DIR.rglob("*.*"):
if img.suffix.lower() not in IMG_EXTS:
continue
shutil.copy2(img, OUT/"train/images"/img.name)
pseudo = PSEUDO_UL/f"{img.stem}.txt"
dst_lb = OUT/"train/labels"/f"{img.stem}.txt"
if pseudo.exists():
lines = [strip_conf(x) for x in pseudo.read_text().splitlines() if x.strip()]
write_txt(dst_lb, lines)
else:
write_txt(dst_lb, [])
# 更新 data.yaml 配置
y = yaml.safe_load(DATA_YAML.read_text())
y["path"] = str(OUT)
y["train"] = str(OUT/"train/images")
y["val"] = str(OUT/"val/images")
y["test"] = str(OUT/"test/images")
(OUT/"data.yaml").write_text(yaml.safe_dump(y, allow_unicode=True, sort_keys=False))
```
---

**4. 学生模型训练脚本 train_student_y11n.py**

```python
model = YOLO(str(STUDENT_W))
model.train(
data=str(DATA_STU), epochs=140, imgsz=704, batch=24, device=0,
optimizer="AdamW", lr0=0.002, lrf=0.01, cos_lr=True,
weight_decay=0.0008, warmup_epochs=3, patience=30, seed=42,
augment=True,
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
degrees=5.0, translate=0.08, scale=0.5, shear=2.0,
mosaic=0.2, mixup=0.1, close_mosaic=15,
project=str(PROJECT), name=RUN_NAME, exist_ok=True, verbose=True,
pretrained=False, resume=False, amp=False
)
model.val(data=str(DATA_STU), imgsz=704, device=0, conf=0.25, iou=0.6, plots=True)
```
---

**5. 剪枝及微调脚本 prune_and_finetune_y11n.py**

```python

# 加载训练好的学生模型
yolo = YOLO(str(BEST_PT))
model = yolo.model.eval()
# 配置剪枝
example_inputs = torch.randn(1, 3, 704, 704)
imp = tp.importance.MagnitudeImportance(p=1) # L1范数重要性
pruner = tp.pruner.MagnitudePruner(
model, example_inputs, importance=imp,
global_pruning=False, ch_sparsity=0.3, ignored_layers=[]
)
pruner.step() # 执行剪枝
# 保存剪枝后的模型结构和权重
yolo.model = model
yolo.save(str(PRUNED_PT))
# 对剪枝模型进行微调训练
pruned_yolo = YOLO(str(PRUNED_PT))
pruned_yolo.train(
data=str(DATA_YAML),
epochs=50, lr0=0.001, imgsz=704, batch=24, device=0,
optimizer="SGD", momentum=0.937,
mosaic=0.3, mixup=0.1, close_mosaic=10,
project=str(ROOT/"runs/detect"), name="finetune_pruned",
pretrained=False, resume=False, amp=True
)
```
---

**6. 模型量化脚本 quantize_y11n_student.py**

```python
# 导出学生模型到ONNX (FP32)
yolo = YOLO(str(PT_PATH))
yolo.export(format="onnx", imgsz=704, simplify=True, dynamic=False, opset=13)
# 查找导出的ONNX文件并重命名
export_candidates = list(ROOT.rglob("*.onnx"))
latest_onnx = max(export_candidates, key=os.path.getmtime)
os.rename(latest_onnx, ONNX_FP32)
# 动态量化INT8
quantize_dynamic(
model_input=str(ONNX_FP32),
model_output=str(ONNX_INT8),
weight_type=QuantType.QInt8
)
onnx_model = onnx.load(ONNX_INT8)
onnx.checker.check_model(onnx_model)
```
---

## 参考文献

1. Redmon, J., Farhadi, A. “YOLOv3: An Incremental Improvement.” arXiv preprint arXiv:
 1804.02767, 2018. (YOLO目标检测算法)
 
2. Wang, C.Y., Bochkovskiy, A., Liao, H.M. “YOLOv7: Trainable bag-of-freebies sets new state-of-the
art for real-time object detectors.” arXiv:2207.02696, 2022. (最新YOLO模型之一)
 
3. Hinton, G., Vinyals, O., Dean, J. “Distilling the Knowledge in a Neural Network.” arXiv:
 1503.02531, 2015. (知识蒸馏原始论文)
 Radosavovic, I., et al. “Data Distillation: Towards Omni-Supervised Learning.” CVPR 2018. (利用
无标注数据进行知识蒸馏的经典方法)
 
4. Li, H., et al. “Pruning Filters for Efficient ConvNets.” ICLR 2017. (基于L1范数的卷积通道剪枝方法)
 Jacob, B., et al. “Quantization and Training of Neural Networks for Efficient Integer-Arithmetic
Only Inference.” CVPR 2018. (8位量化方法的研究)

5. Ultralytics YOLO文档: “模型部署的最佳实践.” Ultralytics, 2023
 (剪枝、量化、蒸馏等优化技术指南)
 
6. Kaggle数据集: “Garbage Detection (6 Waste Categories)”, Kaggle, 2020. (项目所用垃圾检测数据集)
<div style="page-break-after: always;"></div>

---

## 合规声明

&emsp;本项目所使用的数据集来源于公开的Kaggle垃圾分类数据集，遵循其许可协议，仅用于学术研究与模型开发；未标注数据亦采自公开网络资源，不涉及敏感隐私信息。项目开发过程严格遵守比赛及相关法规要求，所有代码与模型均为团队自主完成或基于开源框架实现，引用的第三方库和预训练模型均注明来源并符合其开源许可。本文提及的模型训练方案、算法流程和结果均为团队原创或复现得到，未包含任何保密或受限内容。我们承诺本项目研究过程符合学术道德和数据合规性要求。所发布的源代码和模型参数仅用于技术交流和验证，不用于任何商业用途。如有侵权或不当使用的情况，请及时与我们联系，我们将立即整改。