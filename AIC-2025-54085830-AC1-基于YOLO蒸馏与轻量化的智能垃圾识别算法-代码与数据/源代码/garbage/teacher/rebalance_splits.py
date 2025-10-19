
"""

分层抽样重划 train/valid/test，先写到临时目录，再整体替换，避免源文件被清空。
遇到缺失图片/标签会自动跳过并计数。
"""

import shutil, random
from pathlib import Path
from collections import defaultdict, Counter

random.seed(42)

# ===== 配置 =====
ROOT = Path("/root/autodl-tmp/SWS3009Assg/datasets/kaggle_garbage6")
CLASSES = ["BIODEGRADABLE","CARDBOARD","GLASS","METAL","PAPER","PLASTIC"]
VAL_RATIO, TEST_RATIO = 0.15, 0.10
MIN_VAL_PER_CLASS  = {i: 200 for i in range(len(CLASSES))}
MIN_TEST_PER_CLASS = {i: 120 for i in range(len(CLASSES))}
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
TMP = ROOT / "_tmp_rebalanced"   # 临时目录
# ===============

def list_images(p: Path):
    if not p.exists(): return []
    return [x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS]

def read_cls(lbl_path: Path):
    s = set()
    if not lbl_path.exists(): return s
    try:
        with open(lbl_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                s.add(int(line.split()[0]))
    except Exception:
        return set()
    return s

def collect_all_images(root: Path):
    pools = []
    for sp in ["train","valid","test"]:
        img_dir, lbl_dir = root/sp/"images", root/sp/"labels"
        for im in list_images(img_dir):
            pools.append((im, lbl_dir/(im.stem + ".txt")))
    # 去重（同名跨 split）
    seen, uniq = set(), []
    for im, lb in pools:
        key = im.name.lower()
        if key not in seen:
            seen.add(key)
            uniq.append((im, lb))
    return uniq

def ensure_empty(p: Path):
    if p.exists(): shutil.rmtree(p)
    (p/"train"/"images").mkdir(parents=True, exist_ok=True)
    (p/"train"/"labels").mkdir(parents=True, exist_ok=True)
    (p/"valid"/"images").mkdir(parents=True, exist_ok=True)
    (p/"valid"/"labels").mkdir(parents=True, exist_ok=True)
    (p/"test"/"images").mkdir(parents=True, exist_ok=True)
    (p/"test"/"labels").mkdir(parents=True, exist_ok=True)

def atomic_replace(tmp: Path, root: Path):
    # 先删缓存
    for sp in ["train","valid","test"]:
        cache = root/sp/"labels.cache"
        if cache.exists(): cache.unlink()
    # 删旧目录
    for sp in ["train","valid","test"]:
        for sub in ["images","labels"]:
            d = root/sp/sub
            if d.exists(): shutil.rmtree(d)
    # 拷贝临时内容
    for sp in ["train","valid","test"]:
        for sub in ["images","labels"]:
            src = tmp/sp/sub
            dst = root/sp/sub
            shutil.copytree(src, dst)
    # 删临时目录
    shutil.rmtree(tmp)

def main():
    all_items = collect_all_images(ROOT)
    print(f"[INFO] pooled images: {len(all_items)}")

    # 构建图像信息与每类索引
    img_info, per_cls_imgs = [], defaultdict(list)
    missing_src = 0
    for idx, (im, lb) in enumerate(all_items):
        if not im.exists():
            missing_src += 1
            continue
        s = read_cls(lb)
        img_info.append({"im":im, "lb":lb, "cls":s})
        for c in s:
            per_cls_imgs[c].append(len(img_info)-1)

    if missing_src:
        print(f"[WARN] missing source images skipped: {missing_src}")

    total_cls_imgs = {c: len(per_cls_imgs.get(c, [])) for c in range(len(CLASSES))}
    target_val, target_test = {}, {}
    for c in range(len(CLASSES)):
        t = total_cls_imgs[c]
        val_need  = min(max(int(round(t*VAL_RATIO)),  MIN_VAL_PER_CLASS.get(c,0)), t)
        rem       = t - val_need
        test_need = min(max(int(round(t*TEST_RATIO)), MIN_TEST_PER_CLASS.get(c,0)), rem)
        target_val[c], target_test[c] = val_need, test_need

    print("[TARGET] per-class image counts (val/test):")
    for c in range(len(CLASSES)):
        print(f"  {CLASSES[c]:13s} total={total_cls_imgs[c]:5d}  "
              f"val={target_val[c]:4d}  test={target_test[c]:4d}")

    N = len(img_info)
    assigned = ["unassigned"] * N

    def assign_for_split(name, target_dict):
        need = target_dict.copy()
        while True:
            pending = [(c, need[c]) for c in range(len(CLASSES)) if need[c] > 0]
            if not pending: break
            c = max(pending, key=lambda x: x[1])[0]
            cand = [i for i in per_cls_imgs.get(c, []) if assigned[i]=="unassigned"]
            if not cand:
                need[c] = 0
                continue
            random.shuffle(cand)
            best_idx, best_score = None, 1e9
            for i in cand[:2000]:
                others = img_info[i]["cls"] - {c}
                score = sum(1 for oc in others if need.get(oc,0)>0)
                if score < best_score:
                    best_score, best_idx = score, i
                    if score == 0: break
            assigned[best_idx] = name
            for oc in img_info[best_idx]["cls"]:
                if need.get(oc,0) > 0: need[oc] -= 1

    assign_for_split("valid", target_val)
    assign_for_split("test",  target_test)
    for i in range(N):
        if assigned[i] == "unassigned":
            assigned[i] = "train"

    # 先写到临时目录
    ensure_empty(TMP)
    split_map = {
        "train": (TMP/"train"/"images", TMP/"train"/"labels"),
        "valid": (TMP/"valid"/"images", TMP/"valid"/"labels"),
        "test":  (TMP/"test"/"images",  TMP/"test"/"labels"),
    }
    skipped = 0
    for i,info in enumerate(img_info):
        di, dl = split_map[assigned[i]]
        src_img, src_lbl = info["im"], info["lb"]
        if not src_img.exists(): 
            skipped += 1
            continue
        shutil.copy2(src_img, di/src_img.name)
        if src_lbl.exists():
            shutil.copy2(src_lbl, dl/f"{src_img.stem}.txt")

    print(f"[INFO] copied: {N-skipped}, skipped(missing): {skipped}")
    print("[INFO] assigned counts:", Counter(assigned))

    # 原地替换
    atomic_replace(TMP, ROOT)
    print("[DONE] in-place rebalanced.")
    for sp in ["train","valid","test"]:
        print("  ->", sp, "images:", len(list_images(ROOT/sp/"images")))

if __name__ == "__main__":
    main()
