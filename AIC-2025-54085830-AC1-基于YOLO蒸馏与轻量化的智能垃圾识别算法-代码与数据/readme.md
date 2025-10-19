# 代码与模型数据说明文档

## AIC-2025-54085830-AC1 基于YOLO蒸馏与轻量化的智能垃圾识别算法

---

## 一、总体结构说明

项目目录结构如下：

```plaintext
AIC-2025-54085830-AC1-基于YOLO蒸馏与轻量化的智能垃圾识别算法-代码与数据/
├── 源代码/
│   └── garbage/
│       ├── optimization/        # 模块3: 模型轻量化与部署优化
│       ├── student/             # 模块2: 学生模型蒸馏与训练
│       ├── teacher/             # 模块1: 教师模型训练与伪标签生成
│       └── detect_and_save.py   # 模块4: 推理与检测演示脚本
└── 模型数据/
    └── models/                  # 训练与部署所用权重文件
```

---

## 二、源代码模块说明

（需要修改路径）

### 1️⃣ 教师模型模块 (`teacher` 文件夹)

| 文件名 | 功能说明 |
|:---|:---|
| `train_teacher.py` | 训练 YOLOv11m 教师模型，作为蒸馏知识源。<br>采用 Kaggle 6类垃圾数据集，`epochs=120`，`batch=24`，`optimizer=AdamW`。 |
| `rebalance_splits.py` | 重新划分训练集/验证集/测试集，使各类别样本分布更均衡。 |
| `count_split.py` | 统计每个类别在 `train`/`valid`/`test` 中的图像数量与标注实例数。 |

### 2️⃣ 学生模型模块 (`student` 文件夹)

| 文件名 | 功能说明 |
|:---|:---|
| `gen_pseudo_labels.py` | 使用教师模型 `teacher_best.pt` 对未标注图片生成伪标签（置信度≥0.35），结果保存至 `datasets/unlabeled/pseudo_labels_unlabeled/`。 |
| `build_student_with_unlabeled.py` | 融合原训练集与伪标签数据，构建学生模型训练集 `kaggle_garbage6_student/` 并生成新的 `data.yaml`。 |
| `train_student_y11n.py` | 训练 YOLOv11n 学生模型，使用蒸馏数据，`epochs=140`，`lr=0.002`，`optimizer=AdamW`，模型精度接近教师模型。 |

### 3️⃣ 模型轻量化与部署模块 (`optimization` 文件夹)

| 文件名 | 功能说明 |
|:---|:---|
| `prune_and_finetune_y11n.py` | 对学生模型执行通道剪枝（L1范数重要性，剪除30%通道），再微调50轮恢复精度，生成 `y11n_pruned.pt`。 |
| `quantize_y11n_student.py` | 将剪枝后的学生模型导出为 ONNX，并进行动态量化（FP32→INT8），输出轻量版模型。 |
| `export_fp32_onnx.py` | 手动导出指定权重为 ONNX（FP32），便于验证与跨框架部署。 |

### 4️⃣ 推理与演示模块

| 文件名 | 功能说明 |
|:---|:---|
| `detect_and_save.py` | 通用推理脚本，支持 `.pt` 与 `.onnx` 模型，输入可为视频、摄像头或图片文件夹。自动保存检测结果视频/图片，并输出一张示例帧 `detection_sample.png`。 |
示例：

```
python detect_and_save.py \
  --model ./模型数据/models/y11n_student_int8.onnx \
  --source ./test_video.mp4 \
  --imgsz 704 --conf 0.35 --device cpu \
  --name demo_output
```

---

## 三、模型数据说明

**路径：** `模型数据/models`

| 文件名 | 说明 |
|:---|:---|
| `yolo11m.pt` | YOLOv11m 教师模型初始权重（COCO预训练） |
| `teacher_best.pt` | **训练完成的教师模型（最高精度）** |
| `yolo11n.pt` | YOLOv11n 学生模型初始权重 |
| `student_best.pt` | 蒸馏后学生模型 |
| `y11n_pruned.pt` | 剪枝+微调后的轻量模型 |
| `y11n_student_fp32.onnx` | FP32 ONNX 模型（部署前） |
| `y11n_student_int8.onnx` | **INT8 量化模型（最终部署版本）** |

---

## 四、数据集来源与格式说明

- **主数据集：**
  - Kaggle 开源数据集
  - 🔗 [Garbage Detection (6 Waste Categories)](https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection)

  - 已经给出在模型数据里面
  - 包含六类垃圾： `BIODEGRADABLE`, `CARDBOARD`, `GLASS`, `METAL`, `PAPER`, `PLASTIC`

- **未标注数据：**
  - 可自行从公开网络资源收集的垃圾场景图片，仅用于学术研究和蒸馏，不含隐私信息。


- **数据格式：**
  - 图像尺寸：`704×704`
  - 标签格式：YOLO标准 (`cls cx cy w h`)
  - 伪标签自动生成并合并进学生数据集。

---

## 五、完整执行流程

| 步骤 | 脚本 | 功能 |
|:---:|:---|:---|
| 1️⃣ | `rebalance_splits.py` | 平衡划分 train/valid/test |
| 2️⃣ | `train_teacher.py` | 训练教师模型 YOLOv11m |
| 3️⃣ | `gen_pseudo_labels.py` | 教师模型生成伪标签 |
| 4️⃣ | `build_student_with_unlabeled.py` | 构建学生模型训练集 |
| 5️⃣ | `train_student_y11n.py` | 蒸馏训练学生模型 |
| 6️⃣ | `prune_and_finetune_y11n.py` | 剪枝 + 微调学生模型 |
| 7️⃣ | `quantize_y11n_student.py`+ `export_fp32_onnx.py`| 模型量化 (FP32→INT8) | 
| 8️⃣ | `detect_and_save.py` | 模型推理与结果可视化 |

---

## 六、文件格式与结果输出

| 类型 | 格式 | 示例 |
|:---|:---|:---|
| 模型权重 | `.pt` / `.onnx` | `student_best.pt`, `y11n_student_int8.onnx` |
| 数据标注 | `.txt` | `cls cx cy w h`（归一化坐标） |
| 推理结果 | `.mp4` / `.png` | 输出检测视频及样例图 |
| 数据配置 | `.yaml` | YOLO 数据集配置文件（含路径与类别名） |

---

## 七、环境依赖（推荐）

#### ==== 深度学习核心 ====
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

#### ==== YOLO 主框架 ====
ultralytics==8.3.201        
onnx==1.16.2
onnxruntime==1.17.3

#### ==== 剪枝与轻量化 ====
torch-pruning==1.2.3        
numpy>=1.24.0
opencv-python>=4.9.0
PyYAML>=6.0.1

#### ==== 可视化与辅助 ====
matplotlib
tqdm
pandas






