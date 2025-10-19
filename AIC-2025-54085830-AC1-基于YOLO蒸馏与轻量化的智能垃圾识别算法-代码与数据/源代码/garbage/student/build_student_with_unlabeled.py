# 学生训练集副本
from pathlib import Path
import shutil
import yaml

ROOT = Path("/root/autodl-tmp/SWS3009Assg")
DATASET = ROOT / "datasets" / "kaggle_garbage6"
DATA_YAML = DATASET / "data.yaml"

UNLABELED_DIR = ROOT / "datasets" / "unlabeled" / "images"
PSEUDO_UL = ROOT / "datasets" / "unlabeled" / "pseudo_labels_unlabeled"          # 第一步生成的伪标签
OUT = ROOT / "datasets" / "kaggle_garbage6_student"   

def write_txt(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")

def strip_conf(line: str):
    parts = line.strip().split()
    if len(parts) >= 6:  # cls cx cy w h [conf]
        parts = parts[:5]
    return " ".join(parts)

def main():
    assert DATA_YAML.exists(), "data.yaml 不存在"
    # 重建输出目录
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "train/images").mkdir(parents=True, exist_ok=True)
    (OUT / "train/labels").mkdir(parents=True, exist_ok=True)
    (OUT / "val/images").mkdir(parents=True, exist_ok=True)
    (OUT / "val/labels").mkdir(parents=True, exist_ok=True)
    (OUT / "test/images").mkdir(parents=True, exist_ok=True)
    (OUT / "test/labels").mkdir(parents=True, exist_ok=True)

    # 1) 拷贝 val/test 原样
    for split in ["val", "valid"]:
        if (DATASET / split).exists():  # 你的数据可能叫 valid
            real = split
            break
    else:
        real = "valid"
    for img in (DATASET / real / "images").glob("*"):
        shutil.copy2(img, OUT / "val/images" / img.name)
    for lb in (DATASET / real / "labels").glob("*.txt"):
        shutil.copy2(lb, OUT / "val/labels" / lb.name)
    for img in (DATASET / "test" / "images").glob("*"):
        shutil.copy2(img, OUT / "test/images" / img.name)
    for lb in (DATASET / "test" / "labels").glob("*.txt"):
        shutil.copy2(lb, OUT / "test/labels" / lb.name)

    # 2) 训练集（strict）：先拷贝 train/images；labels 仅复制 GT
    for img in (DATASET / "train" / "images").glob("*"):
        shutil.copy2(img, OUT / "train/images" / img.name)
        gt = DATASET / "train" / "labels" / (img.stem + ".txt")
        dst = OUT / "train" / "labels" / (img.stem + ".txt")
        if gt.exists():
            shutil.copy2(gt, dst)
        else:
            dst.write_text("", encoding="utf-8")  # 无GT写空

    # 3) 追加 unlabeled + pseudo：把未标注图复制进 train/images，并用伪标签写入 train/labels
    assert UNLABELED_DIR.exists(), f"未找到 unlabeled/images: {UNLABELED_DIR}"
    ul_imgs = list(UNLABELED_DIR.rglob("*.*"))
    print(f"[INFO] add unlabeled images: {len(ul_imgs)}")
    for img in ul_imgs:
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        dst_img = OUT / "train/images" / img.name
        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        pseudo = PSEUDO_UL / (img.stem + ".txt")
        dst_lb = OUT / "train/labels" / (img.stem + ".txt")
        if pseudo.exists():
            lines = [strip_conf(x) for x in pseudo.read_text(encoding="utf-8").splitlines() if x.strip()]
            write_txt(dst_lb, lines)
        else:
            # 没有伪标签就写空；训练时这张图相当于无目标背景
            write_txt(dst_lb, [])

    # 4) 生成新的 data.yaml
    y = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    # 统一改为绝对/明确路径（避免相对路径问题）
    y["path"] = str(OUT)
    y["train"] = str(OUT / "train" / "images")
    y["val"] = str(OUT / "val" / "images")
    y["test"] = str(OUT / "test" / "images")
    (OUT / "data.yaml").write_text(yaml.safe_dump(y, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[OK] student dataset ready -> {OUT}")

if __name__ == "__main__":
    main()
