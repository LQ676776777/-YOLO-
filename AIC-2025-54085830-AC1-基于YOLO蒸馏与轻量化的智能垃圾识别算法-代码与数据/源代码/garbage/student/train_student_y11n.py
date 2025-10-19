
from pathlib import Path
from ultralytics import YOLO
import os

ROOT = Path("/root/autodl-tmp/SWS3009Assg")
STUDENT_W = ROOT / "models" / "yolo11n.pt"
DATA_STU  = ROOT / "datasets" / "kaggle_garbage6_student" / "data.yaml"
PROJECT   = ROOT / "runs" / "detect"
RUN_NAME  = "train_student"

def rm_cache(data_yaml: Path):
    base = data_yaml.parent
    for split in ["train", "val", "test"]:
        c = base / split / "labels.cache"
        if c.exists():
            c.unlink()

def main():
    # 禁用联网检查（不用 SETTINGS）
    os.environ["ULTRALYTICS_HUB"] = "0"
    os.environ["ULTRALYTICS_CHECKS"] = "0"

    assert STUDENT_W.exists(), f"权重未找到: {STUDENT_W}"
    assert STUDENT_W.stat().st_size > 1_000_000, "权重文件可能损坏或过小"

    rm_cache(Path(DATA_STU))

    print(f"[INFO] Loading local weights: {STUDENT_W}")
    model = YOLO(str(STUDENT_W))

    model.train(
        data=str(DATA_STU),
        epochs=140, imgsz=704, batch=24, device=0,
        optimizer="AdamW", lr0=0.002, lrf=0.01, cos_lr=True,
        weight_decay=0.0008, warmup_epochs=3, patience=30, seed=42,
        augment=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.08, scale=0.5, shear=2.0,
        mosaic=0.2, mixup=0.1, close_mosaic=15,
        project=str(PROJECT), name=RUN_NAME, exist_ok=True, verbose=True,
        pretrained=False,   # 只用本地权重
        resume=False,
        amp=False           # 关闭 AMP 
    )

    model.val(
        data=str(DATA_STU), imgsz=704, device=0,
        conf=0.25, iou=0.6, plots=True
    )

if __name__ == "__main__":
    main()
