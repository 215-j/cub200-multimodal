"""01_singleModal_ImageOnly.py —— 单模态基线（只用图像）。

与 02_singleModal_TextOnly.py 对应，是「图文多模态」实验的图像侧参照。
做法：ResNet18（ImageNet 预训练）去掉原 1000 类 fc，换成 100 类分类头，
在 CUB 子集上微调，用 test 集报告 Accuracy / 宏平均 Precision / Recall / F1。

注意：训练/测试 transform 不同 —— 训练带随机裁剪与翻转（数据增强），
测试只用 resize + 中心裁剪，保证评估可复现。

运行：python 01_singleModal_ImageOnly.py
"""
import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ======================
# 参数
# ======================
root_dir = "datasets"      # 标注 json 目录
image_root = "raw"         # 图片根目录（标注里的 image 字段相对它）
results_dir = "results"

backbone = "resnet18"
num_classes = 100
batch_size = 16
epochs = 12
lr = 1e-4
weight_decay = 1e-4
image_size = 224
seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


# ======================
# Dataset
# ======================
class ImageBirdDataset(Dataset):
    """读 annotations_*.json，返回 (图像张量, 标签)。"""

    def __init__(self, json_file, image_root, transform):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = data["annotations"]
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        img_path = os.path.join(self.image_root, item["image"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(item["label"], dtype=torch.long)
        return image, label


# ======================
# transforms
# ======================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ======================
# 模型
# ======================
def build_model():
    """ImageNet 预训练 ResNet18，替换分类头为 100 类。"""
    if backbone == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif backbone == "resnet34":
        net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"不支持的 backbone: {backbone}")
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


# ======================
# 评估工具
# ======================
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu()
            y_pred.extend(preds.numpy())
            y_true.extend(labels.numpy())
    return y_true, y_pred


def report(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def main():
    torch.manual_seed(seed)

    train_json = os.path.join(root_dir, "annotations_train.json")
    val_json = os.path.join(root_dir, "annotations_val.json")
    test_json = os.path.join(root_dir, "annotations_test.json")

    train_set = ImageBirdDataset(train_json, image_root, train_transform)
    val_set = ImageBirdDataset(val_json, image_root, eval_transform)
    test_set = ImageBirdDataset(test_json, image_root, eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print("train/val/test:", len(train_set), len(val_set), len(test_set))

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state = None
    history = []

    print("start training...")
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        yv, pv = evaluate(model, val_loader)
        val_acc = accuracy_score(yv, pv)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch + 1, "loss": total_loss / len(train_loader),
                        "train_acc": train_acc, "val_acc": val_acc})
        print(f"Epoch [{epoch+1}/{epochs}] loss={total_loss/len(train_loader):.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"({time.time()-t0:.1f}s)")

    # 用验证集最好的模型去测 test
    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = evaluate(model, test_loader)
    metrics = report(y_true, y_pred)

    print("====================")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"宏平均精确率 (Precision): {metrics['precision_macro']:.4f}")
    print(f"宏平均召回率 (Recall): {metrics['recall_macro']:.4f}")
    print(f"宏平均 F1 分数 (F1-Score): {metrics['f1_macro']:.4f}")
    print(f"best_val_acc: {best_val_acc:.4f}")
    print("====================")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "01_image_only.json"), "w", encoding="utf-8") as f:
        json.dump({"config": {"backbone": backbone, "epochs": epochs,
                              "batch_size": batch_size, "lr": lr,
                              "image_size": image_size},
                   "best_val_acc": best_val_acc,
                   "test": metrics,
                   "history": history}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
