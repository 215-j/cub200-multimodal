"""00_prepare_data.py —— 数据准备：把 raw/ 整理成统一的 json 标注。

原始数据（解压 CUB_200_subset.zip 得到）
------------------------------------
raw/
├── images/<类别目录>/<图片名>.jpg      # 100 类，共 5871 张
└── text/<类别目录>/<图片名>.txt        # 每个 txt 是该图对应的 10 条 caption（每行一条）

本脚本做的事
------------
1. 扫描 raw/images 得到 100 个类别，按目录名排序后编号 label = 0..99；
2. 对每张图去找同名的 text/<类>/<图名>.txt，读入 10 条 caption；
   （缺 txt 的图片直接跳过，保证图文成对）
3. 按类别做分层切分 train / val / test = 70% / 15% / 15%（同一类别内部切分，
   保证每个类别在三份数据里都出现，固定随机种子保证可复现）；
4. 输出：
      datasets/annotations_train.json
      datasets/annotations_val.json
      datasets/annotations_test.json
      datasets/classes.json
   标注格式（与 02_singleModal_TextOnly.py 期望的一致）：
      {"annotations": [
          {"image": "images/001.Black_footed_Albatross/xxx.jpg",
           "label": 0,
           "captions": ["...", "...", ...]},
          ...
      ]}
   "image" 是相对 raw/ 的路径，画图时用 os.path.join(IMAGE_ROOT, image) 即可。

运行：python 00_prepare_data.py
"""
import os
import json
import random
from collections import OrderedDict

# ======================
# 参数
# ======================
RAW_DIR = "raw"            # 解压后的原始数据目录
OUT_DIR = "datasets"       # 生成的标注目录（02 脚本默认读这里）
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
SEED = 42

IMG_EXTS = (".jpg", ".jpeg", ".png")


def build_split_of_class(items, train_ratio, val_ratio, rng):
    """把一个类别内部的样本按比例切成 train/val/test（先打乱再切）。"""
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def main():
    images_dir = os.path.join(RAW_DIR, "images")
    text_dir = os.path.join(RAW_DIR, "text")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"找不到 {images_dir}，请先解压 CUB_200_subset.zip 到 {RAW_DIR}/"
        )

    # 1. 类别列表：按目录名排序，保证 label 编号稳定
    classes = sorted(
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[data] 类别数 = {len(classes)}")

    splits = OrderedDict(train=[], val=[], test=[])
    rng = random.Random(SEED)          # 全局种子，逐类别复用
    n_missing_text = 0
    per_class = []

    # 2. 逐类别收集「图 + 10 条 caption」
    for cls in classes:
        img_files = sorted(
            f for f in os.listdir(os.path.join(images_dir, cls))
            if f.lower().endswith(IMG_EXTS)
        )
        items = []
        for f in img_files:
            stem = os.path.splitext(f)[0]
            txt_path = os.path.join(text_dir, cls, stem + ".txt")
            if not os.path.exists(txt_path):
                n_missing_text += 1
                continue
            with open(txt_path, "r", encoding="utf-8") as fh:
                captions = [ln.strip() for ln in fh if ln.strip()]
            if not captions:                # 空标注也跳过
                n_missing_text += 1
                continue
            items.append({
                "image": f"images/{cls}/{f}",
                "label": class_to_idx[cls],
                "captions": captions,
            })
        tr, va, te = build_split_of_class(items, TRAIN_RATIO, VAL_RATIO, rng)
        splits["train"] += tr
        splits["val"] += va
        splits["test"] += te
        per_class.append((cls, len(items), len(tr), len(va), len(te)))

    # 3. 再整体打乱一次，避免按类别顺序排列（DataLoader shuffle 之前更均匀）
    for k in splits:
        rng.shuffle(splits[k])

    # 4. 落盘
    os.makedirs(OUT_DIR, exist_ok=True)
    for k, v in splits.items():
        path = os.path.join(OUT_DIR, f"annotations_{k}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"annotations": v}, fh, ensure_ascii=False, indent=2)
        print(f"[data] {path}: {len(v)} 条")

    with open(os.path.join(OUT_DIR, "classes.json"), "w", encoding="utf-8") as fh:
        json.dump(classes, fh, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in splits.values())
    print(f"[data] 总计 {total} 条（跳过无 caption 的图 {n_missing_text} 张）")
    print("[data] 每类样本数 (类别, 总数, train, val, test) 示例：")
    for row in per_class[:3] + per_class[-1:]:
        print("       ", row)


if __name__ == "__main__":
    main()
