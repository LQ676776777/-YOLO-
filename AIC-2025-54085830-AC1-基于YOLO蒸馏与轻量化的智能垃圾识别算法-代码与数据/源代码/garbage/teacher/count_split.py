# 验证类别平衡
from pathlib import Path
from collections import defaultdict

ROOT = Path("/root/autodl-tmp/SWS3009Assg/datasets/kaggle_garbage6")
SPLITS = ["train", "valid", "test"]
CLSN = ["BIODEGRADABLE","CARDBOARD","GLASS","METAL","PAPER","PLASTIC"]

def read_lbl(p):
    try:
        with open(p) as f:
            return [int(l.split()[0]) for l in f if l.strip()]
    except FileNotFoundError:
        return []

for sp in SPLITS:
    img_dir = ROOT/sp/"images"
    lbl_dir = ROOT/sp/"labels"
    per_cls_imgs = [set() for _ in CLSN]
    per_cls_inst = [0]*len(CLSN)
    total_imgs = 0

    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in {".jpg",".jpeg",".png",".bmp",".webp"}: 
            continue
        total_imgs += 1
        lbl = lbl_dir/f"{img.stem}.txt"
        ids = read_lbl(lbl)
        for cid in ids:
            per_cls_imgs[cid].add(img.name)
            per_cls_inst[cid] += 1

    print(f"\n== {sp.upper()} ==")
    print("Total images:", total_imgs)
    for i,n in enumerate(CLSN):
        print(f"{n:13s}  Images={len(per_cls_imgs[i]):5d}  Instances={per_cls_inst[i]:5d}")
