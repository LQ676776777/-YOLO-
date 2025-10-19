from ultralytics import YOLO
import torch
import torch_pruning as tp
from pathlib import Path
import os
# 修改路径区域
ROOT = Path("/root/autodl-tmp/SWS3009Assg")
BEST_PT = ROOT / "models/student_best.pt"
DATA_YAML = ROOT / "datasets/kaggle_garbage6/data.yaml"
PRUNED_PT = ROOT / "models/y11n_pruned.pt"

os.makedirs(ROOT / "models", exist_ok=True)

# ====== 加载模型 ======
print("[INFO] Loading student model ...")
yolo = YOLO(str(BEST_PT))
model = yolo.model.eval()

# ====== 剪枝配置 ======
example_inputs = torch.randn(1, 3, 704, 704)
imp = tp.importance.MagnitudeImportance(p=1)  # L1范数重要性
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    global_pruning=False,   # 按层剪
    ch_sparsity=0.3,        # 剪除比例
    ignored_layers=[],
)

print("[INFO] Starting pruning with torch-pruning 1.2.x API ...")
pruner.step()  # 执行剪枝

# ====== 保存剪枝模型 ======

print("[INFO] Exporting pruned model as YOLO checkpoint ...")
yolo.model = model  # 替换为剪枝后的模型结构
yolo.save(str(PRUNED_PT))  # 用 Ultralytics 自带保存函数
print(f"[OK] YOLO checkpoint saved → {PRUNED_PT}")


# ====== 微调训练 ======
print("[INFO] Fine-tuning pruned model ...")
pruned_yolo = YOLO(str(PRUNED_PT))
pruned_yolo.train(
    data=str(DATA_YAML),
    epochs=50,
    lr0=0.001,
    imgsz=704,
    batch=24,
    device=0,
    optimizer="SGD",
    momentum=0.937,
    mosaic=0.3,
    mixup=0.1,
    close_mosaic=10,
    project=str(ROOT / "runs" / "detect"),
    name="finetune_pruned",
    pretrained=False,
    resume=False,
    amp=True
)

print("[DONE] Fine-tuning complete ✅")
