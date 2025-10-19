# # 在线下载 -*-
# from pathlib import Path
# from ultralytics import YOLO
# import os

# ROOT = Path("/root/autodl-tmp/SWS3009Assg")
# STUDENT_W = ROOT / "models" / "yolo11n.pt"
# DATA_STU  = ROOT / "datasets" / "kaggle_garbage6_student" / "data.yaml"
# PROJECT   = ROOT / "runs" / "detect"
# RUN_NAME  = "train_student"

# def rm_cache(data_yaml: Path):
#     base = data_yaml.parent
#     for split in ["train", "val", "test"]:
#         c = base / split / "labels.cache"
#         if c.exists():
#             c.unlink()

# def main():
#     # 1) 禁用联网检查（仅环境变量即可，不用 SETTINGS）
#     os.environ["ULTRALYTICS_HUB"] = "0"
#     os.environ["ULTRALYTICS_CHECKS"] = "0"

#     # 2) 确认本地权重存在
#     assert STUDENT_W.exists(), f"权重未找到: {STUDENT_W}"
#     assert STUDENT_W.stat().st_size > 1_000_000, "权重文件可能损坏或过小"

#     # 3) 清理 labels.cache
#     rm_cache(Path(DATA_STU))

#     # 4) 用本地权重初始化
#     print(f"[INFO] Loading local weights: {STUDENT_W}")
#     model = YOLO(str(STUDENT_W))

#     # 5) 训练（关键：pretrained=False, amp=False）
#     model.train(
#         data=str(DATA_STU),
#         epochs=140, imgsz=704, batch=24, device=0,
#         optimizer="AdamW", lr0=0.002, lrf=0.01, cos_lr=True,
#         weight_decay=0.0008, warmup_epochs=3, patience=30, seed=42,
#         augment=True,
#         hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
#         degrees=5.0, translate=0.08, scale=0.5, shear=2.0,
#         mosaic=0.2, mixup=0.1, close_mosaic=15,
#         project=str(PROJECT), name=RUN_NAME, exist_ok=True, verbose=True,
#         pretrained=False,   # ✅ 只用本地权重，不再下载
#         resume=False,
#         amp=False           # ✅ 禁用 AMP 自检，根因解决
#     )

#     # 6) 验证
#     model.val(
#         data=str(DATA_STU), imgsz=704, device=0,
#         conf=0.25, iou=0.6, plots=True
#     )

# if __name__ == "__main__":
#     main()

# -*- 离线 -*-
from pathlib import Path
from ultralytics import YOLO
import os

# ==============================
# 路径配置
# ==============================
ROOT = Path("/root/autodl-tmp/SWS3009Assg")
TEACHER_W = ROOT / "models" / "yolo11m.pt"   # 教师模型权重
DATA_DIR  = ROOT / "datasets" / "kaggle_garbage6"
DATA_YAML = DATA_DIR / "data.yaml"
PROJECT   = ROOT / "runs" / "detect"
RUN_NAME  = "train_teacher111"

# ==============================
# 工具函数：清除标签缓存
# ==============================
def rm_cache(data_dir: Path):
    for split in ["train", "valid", "test"]:
        c = data_dir / split / "labels.cache"
        if c.exists():
            c.unlink()

# ==============================
# 主程序
# ==============================
def main():
    # 1) 禁用联网检查（仅环境变量即可，不用 SETTINGS）
    os.environ["ULTRALYTICS_HUB"] = "0"
    os.environ["ULTRALYTICS_CHECKS"] = "0"

    # 2) 确认本地权重存在
    assert TEACHER_W.exists(), f"权重未找到: {TEACHER_W}"
    assert TEACHER_W.stat().st_size > 1_000_000, "权重文件可能损坏或过小"

    # 3) 清理 labels.cache
    rm_cache(DATA_DIR)

    # 4) 用本地权重初始化
    print(f"[INFO] Loading local weights: {TEACHER_W}")
    model = YOLO(str(TEACHER_W))

    # 5) 训练
    model.train(
        data=str(DATA_YAML),
        epochs=120, imgsz=704, batch=24, device=0,
        optimizer="AdamW", lr0=0.01, lrf=0.01, cos_lr=True,
        weight_decay=5e-4, warmup_epochs=3, patience=30, seed=42,
        augment=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.08, scale=0.5, shear=2.0,
        mosaic=0.5, mixup=0.0, close_mosaic=15,
        project=str(PROJECT), name=RUN_NAME, exist_ok=True, verbose=True,
        pretrained=False,   # ✅ 本地权重
        resume=False,
        amp=False           # ✅ 禁用 AMP 自检
    )

    # 6) 用 best.pt 分别在 valid 和 test 上验证
    best = ROOT / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"
    assert best.exists(), f"未找到 best.pt: {best}"

    print(f"[VAL] evaluating on valid split with {best}")
    YOLO(str(best)).val(
        data=str(DATA_YAML), split="val", device=0, plots=True,
        project=str(PROJECT), name="train_teacher_val", exist_ok=True
    )

    print(f"[TEST] evaluating on test split with {best}")
    YOLO(str(best)).val(
        data=str(DATA_YAML), split="test", device=0, plots=True,
        project=str(PROJECT), name="train_teacher_test", exist_ok=True
    )

# ==============================
# 入口
# ==============================
if __name__ == "__main__":
    main()
