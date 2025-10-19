
from pathlib import Path
from ultralytics import YOLO
import shutil

ROOT = Path("/root/autodl-tmp/SWS3009Assg")
UNLABELED = ROOT / "datasets" / "unlabeled" / "images"   # 有子文件夹
TEACHER = ROOT / "models" / "teacher_best.pt"

OUT_DIR = ROOT / "runs" / "detect" / "pseudo_unlabeled"
SAVE_PL = ROOT / "datasets" / "unlabeled" / "pseudo_labels_unlabeled" 

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def count_images_recursively(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)

def main():
    assert UNLABELED.exists(), f"unlabeled/images 不存在: {UNLABELED}"
    n = count_images_recursively(UNLABELED)
    print(f"[INFO] will scan recursively: {UNLABELED}  images={n}")
    if n == 0:
        raise SystemExit("未在 unlabeled/images 子目录中发现图片，请检查后缀与路径。")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    SAVE_PL.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(TEACHER))
    model.predict(
        source=str(UNLABELED / "**" / "*"),  # ★ 递归匹配所有子目录
        conf=0.35, iou=0.60,
        imgsz=704, half=True, device=0,
        save_txt=True, save_conf=True,
        project=str(ROOT / "runs" / "detect"),
        name="pseudo_unlabeled",
        exist_ok=True,
        max_det=300, batch=24, stream=False, verbose=True
    )

    # 汇总 labels（Ultralytics会按子目录产出）
    moved = 0
    for p in OUT_DIR.rglob("*.txt"):
        (SAVE_PL / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        moved += 1
    print(f"[OK] moved pseudo labels: {moved} -> {SAVE_PL}")

if __name__ == "__main__":
    main()
