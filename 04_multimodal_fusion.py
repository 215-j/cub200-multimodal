"""04_multimodal_fusion.py —— 多模态融合：图文一起预测鸟类（100 类）。

三种融合方式（--mode 选择，或用 all 依次跑完并汇总）：
  early      早融合：图像全局向量 [B,d] 与文本向量 [B,d] 拼接 -> MLP 分类头
  late       晚融合：两个模态各自出一个 100 维 logits，再相加（决策级融合）
  attention  注意力融合：文本向量当 query，图像的 49 个空间 token 当 key/value
             做跨模态注意力（把图像每个空间位置当成一个"视觉词"）-> 分类头

编码器直接复用 03_image_encoder.py / 03_text_encoder.py。
这两个文件名以数字开头，不能用 import 语句导入，所以这里用 importlib 按路径加载。

CPU 友好设置：默认 freeze=True，只训练两侧投影层 + 融合层 + 分类头，
主干（ResNet18 / BERT）不更新参数，速度最快。

运行示例：
    python 04_multimodal_fusion.py --mode early
    python 04_multimodal_fusion.py --mode all      # 三种都跑 + 汇总
"""
import os
import json
import time
import argparse
import importlib.util
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_module(path, name):
    """按文件路径加载模块（解决文件名以数字开头无法 import 的问题）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


image_encoder_mod = _load_module(os.path.join(HERE, "03_image_encoder.py"), "image_encoder")
text_encoder_mod = _load_module(os.path.join(HERE, "03_text_encoder.py"), "text_encoder")
ImageEncoder = image_encoder_mod.ImageEncoder
TextEncoder = text_encoder_mod.TextEncoder
load_tokenizer = text_encoder_mod.load_tokenizer


# ======================
# Dataset
# ======================
class FusionBirdDataset(Dataset):
    """返回 (图像, input_ids, attention_mask, label)。

    每张图有 10 条 caption，训练时随机取一条（和 02 脚本同样的做法），
    增加文本侧的多样性；测试时会用固定策略（默认取第一条）保证可复现。
    """

    def __init__(self, json_file, image_root, tokenizer, transform,
                 max_length=64, random_caption=True):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data["annotations"]
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.random_caption = random_caption

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        img_path = os.path.join(self.image_root, item["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        captions = item["captions"]
        if self.random_caption:
            text = captions[torch.randint(0, len(captions), (1,)).item()]
        else:
            text = captions[0]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "image": image,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


# ======================
# 融合模型
# ======================
def mlp_head(in_dim, num_classes, hidden=256, dropout=0.3):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, num_classes),
    )


class FusionModel(nn.Module):
    def __init__(self, mode, num_classes=100, dim=256, backbone="resnet18",
                 model_name="bert-base-uncased", freeze=True,
                 num_heads=4, dropout=0.3):
        super().__init__()
        assert mode in ("early", "late", "attention")
        self.mode = mode
        self.img_enc = ImageEncoder(backbone=backbone, pretrained=True,
                                    freeze=freeze, out_dim=dim, dropout=0.1)
        self.txt_enc = TextEncoder(model_name=model_name, pretrained=True,
                                   freeze=freeze, out_dim=dim, pooling="mean",
                                   dropout=0.1)
        self.dim = dim

        if mode == "early":
            # 拼接后过 MLP
            self.head = mlp_head(2 * dim, num_classes, hidden=2 * dim, dropout=dropout)
        elif mode == "late":
            # 两模态各自分类，再把 logits 相加（也可改成加权/平均概率）
            self.img_head = mlp_head(dim, num_classes, hidden=dim, dropout=dropout)
            self.txt_head = mlp_head(dim, num_classes, hidden=dim, dropout=dropout)
        else:
            # 文本 query 对图像 token 做跨模态注意力
            self.attn = nn.MultiheadAttention(dim, num_heads=num_heads,
                                              dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(dim)
            self.head = mlp_head(dim, num_classes, hidden=dim, dropout=dropout)

    def forward(self, images, input_ids, attention_mask):
        if self.mode == "attention":
            img_tokens = self.img_enc(images, return_tokens=True)      # [B, 49, d]
            query = self.txt_enc(input_ids, attention_mask).unsqueeze(1)  # [B, 1, d]
            attn_out, _ = self.attn(query, img_tokens, img_tokens)     # [B, 1, d]
            fused = self.norm(query + attn_out).squeeze(1)             # 残差 + LN
            return self.head(fused)

        img_vec = self.img_enc(images)                                 # [B, d]
        txt_vec = self.txt_enc(input_ids, attention_mask)              # [B, d]

        if self.mode == "early":
            return self.head(torch.cat([img_vec, txt_vec], dim=1))
        # late：logits 相加
        return self.img_head(img_vec) + self.txt_head(txt_vec)


# ======================
# transforms
# ======================
def build_transforms(image_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def report(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_one(mode, args, device):
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.model_name)
    train_tf, eval_tf = build_transforms(args.image_size)

    train_json = os.path.join(args.root_dir, "annotations_train.json")
    val_json = os.path.join(args.root_dir, "annotations_val.json")
    test_json = os.path.join(args.root_dir, "annotations_test.json")

    train_set = FusionBirdDataset(train_json, args.image_root, tokenizer, train_tf,
                                  args.max_length, random_caption=True)
    val_set = FusionBirdDataset(val_json, args.image_root, tokenizer, eval_tf,
                                args.max_length, random_caption=False)
    test_set = FusionBirdDataset(test_json, args.image_root, tokenizer, eval_tf,
                                 args.max_length, random_caption=False)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FusionModel(mode, num_classes=args.num_classes, dim=args.dim,
                        backbone=args.backbone, model_name=args.model_name,
                        freeze=args.freeze, dropout=args.dropout).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[{mode}] 可训练参数 {n_trainable/1e6:.2f}M / 总 {n_total/1e6:.2f}M "
          f"(freeze={args.freeze})")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    def evaluate(loader):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(images, ids, mask)
                preds = torch.argmax(logits, dim=1).cpu()
                y_pred.extend(preds.numpy())
                y_true.extend(batch["label"].numpy())
        return y_true, y_pred

    best_val_acc, best_state, history = 0.0, None, []
    print(f"[{mode}] start training...")
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images, ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        yv, pv = evaluate(val_loader)
        val_acc = accuracy_score(yv, pv)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch + 1, "loss": total_loss / len(train_loader),
                        "train_acc": train_acc, "val_acc": val_acc})
        print(f"[{mode}] Epoch [{epoch+1}/{args.epochs}] "
              f"loss={total_loss/len(train_loader):.4f} train_acc={train_acc:.4f} "
              f"val_acc={val_acc:.4f} ({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_pred = evaluate(test_loader)
    metrics = report(y_true, y_pred)

    print("====================")
    print(f"[{mode}] 准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"[{mode}] 宏平均精确率 (Precision): {metrics['precision_macro']:.4f}")
    print(f"[{mode}] 宏平均召回率 (Recall): {metrics['recall_macro']:.4f}")
    print(f"[{mode}] 宏平均 F1 分数 (F1-Score): {metrics['f1_macro']:.4f}")
    print(f"[{mode}] best_val_acc: {best_val_acc:.4f}")
    print("====================")

    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, f"04_fusion_{mode}.json"),
              "w", encoding="utf-8") as f:
        json.dump({"config": {"mode": mode, "backbone": args.backbone,
                              "model_name": args.model_name, "dim": args.dim,
                              "freeze": args.freeze, "epochs": args.epochs,
                              "batch_size": args.batch_size, "lr": args.lr},
                   "best_val_acc": best_val_acc,
                   "test": metrics, "history": history},
                  f, ensure_ascii=False, indent=2)
    return mode, metrics, best_val_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="early",
                   choices=["early", "late", "attention", "all"])
    p.add_argument("--root_dir", default="datasets")
    p.add_argument("--image_root", default="raw")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=100)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--freeze", action="store_true", default=True)
    p.add_argument("--no_freeze", dest="freeze", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    modes = ["early", "late", "attention"] if args.mode == "all" else [args.mode]
    summary = {}
    for m in modes:
        _, metrics, best_val = run_one(m, args, device)
        summary[m] = {"best_val_acc": best_val, **metrics}

    if len(modes) > 1:
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, "04_fusion_summary.json"),
                  "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("\n==================== 融合方式对比 ====================")
        print(f"{'mode':<12}{'acc':>8}{'prec':>8}{'recall':>8}{'f1':>8}")
        for m, v in summary.items():
            print(f"{m:<12}{v['accuracy']:>8.4f}{v['precision_macro']:>8.4f}"
                  f"{v['recall_macro']:>8.4f}{v['f1_macro']:>8.4f}")


if __name__ == "__main__":
    main()
