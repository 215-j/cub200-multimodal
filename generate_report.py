"""生成实验报告：训练曲线图 (matplotlib) + Word 文档 (python-docx)。"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn

ROOT = r"E:\计算机视觉2"
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "实验报告.docx")
FIG = os.path.join(RESULTS, "training_curves.png")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 读数据 ----------------
j01 = json.load(open(os.path.join(RESULTS, "01_image_only.json")))
fusions = {m: json.load(open(os.path.join(RESULTS, f"04_fusion_{m}.json")))
           for m in ["early", "late", "attention"]}

# 02 文本的每轮数据从 log 解析（它没有保存 val_acc，只有 train）
log02 = open(os.path.join(RESULTS, "02_text_only.log"), encoding="utf-8").read()
h02 = [{"epoch": i + 1,
        "loss": float(m.group(1)),
        "train_acc": float(m.group(2))}
       for i, m in enumerate(re.finditer(r"epoch: \d+ loss: ([\d.]+) Train Accuracy: ([\d.]+)", log02))]

metrics02 = {"accuracy": 0.3516, "precision_macro": 0.3495,
             "recall_macro": 0.3505, "f1_macro": 0.3281}

rows = [
    ("图像单模态 (ResNet18 微调)", j01["best_val_acc"], j01["test"]),
    ("文本单模态 (BERT 微调)", None, metrics02),
    ("多模态·早融合 (Early)", fusions["early"]["best_val_acc"], fusions["early"]["test"]),
    ("多模态·晚融合 (Late)", fusions["late"]["best_val_acc"], fusions["late"]["test"]),
    ("多模态·注意力融合 (Attention)", fusions["attention"]["best_val_acc"], fusions["attention"]["test"]),
]

# ---------------- 画图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

ax = axes[0]
ax.plot([h["epoch"] for h in j01["history"]], [h["loss"] for h in j01["history"]],
        "o-", label="01 图像", color="#1f77b4")
ax.plot([h["epoch"] for h in h02], [h["loss"] for h in h02],
        "s-", label="02 文本", color="#ff7f0e")
for m, c, mk in [("early", "#2ca02c", "^"), ("late", "#d62728", "v"), ("attention", "#9467bd", "D")]:
    h = fusions[m]["history"]
    ax.plot([x["epoch"] for x in h], [x["loss"] for x in h], mk + "-", label=f"04 {m}", color=c)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("训练损失曲线")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot([h["epoch"] for h in j01["history"]], [h["val_acc"] for h in j01["history"]],
        "o-", label="01 图像 (val)", color="#1f77b4")
ax.plot([h["epoch"] for h in j01["history"]], [h["train_acc"] for h in j01["history"]],
        "o--", alpha=0.5, label="01 图像 (train)", color="#1f77b4")
ax.plot([h["epoch"] for h in h02], [h["train_acc"] for h in h02],
        "s-", label="02 文本 (train)", color="#ff7f0e")
for m, c, mk in [("early", "#2ca02c", "^"), ("late", "#d62728", "v"), ("attention", "#9467bd", "D")]:
    h = fusions[m]["history"]
    ax.plot([x["epoch"] for x in h], [x["val_acc"] for x in h], mk + "-", label=f"04 {m} (val)", color=c)
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("准确率曲线")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(FIG, bbox_inches="tight")
plt.close(fig)
print("figure saved:", FIG)

# ---------------- Word 报告 ----------------
doc = Document()

# 中文字体设置
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(11)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")


def set_cn(run, font="宋体"):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)


def heading(text, level=1):
    h = doc.add_heading("", level=level)
    r = h.add_run(text)
    set_cn(r, "黑体")
    r.font.color.rgb = RGBColor(0, 0, 0)
    return h


def para(text, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_cn(r)
    r.bold = bold
    return p


title = doc.add_heading("", level=0)
tr = title.add_run("基于图像与文本多模态融合的细粒度鸟类分类实验报告")
set_cn(tr, "黑体")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

para("实验日期：2026-10-13    实验环境：CPU（24 线程），PyTorch 2.4.1，Python 3.8")

# ========== 1 实验目的 ==========
heading("一、实验目的")
para("本实验在 CUB-200 鸟类数据集子集（100 类）上，分别构建图像单模态、文本单模态基线模型，"
     "并在此基础上实现早融合（Early Fusion）、晚融合（Late Fusion）、注意力融合（Attention Fusion）"
     "三种图文多模态融合策略，比较五种方法在 100 类细粒度鸟类分类任务上的性能差异，"
     "分析不同模态的贡献以及融合方式对分类效果的影响。")

# ========== 2 数据集 ==========
heading("二、数据集")
para("数据集为 CUB-200-2011 的子集，共 100 个鸟类类别，5 871 张图像。"
     "每张图像配有 10 条英文描述文本（caption），描述鸟的羽色、喙形、翅膀等外观特征。")
para("数据预处理：对每张图像读取其对应的 caption 文件（其中 7 张图像缺少文本标注，已剔除），"
     "最终保留 5 864 个有效样本。按类别分层随机划分为训练集 / 验证集 / 测试集 = 70% / 15% / 15%，"
     "随机种子固定为 42，得到训练集 4 093 条、验证集 858 条、测试集 913 条。")
para("图像预处理：训练阶段采用 RandomResizedCrop(224) + 随机水平翻转进行数据增强；"
     "验证与测试阶段采用 Resize(256) + CenterCrop(224)；均使用 ImageNet 均值方差标准化。"
     "文本预处理：使用 BERT tokenizer 分词，最大长度 64，训练时从 10 条 caption 中随机采样一条。")

# ========== 3 方法 ==========
heading("三、模型与方法")

heading("3.1 图像单模态基线", 2)
para("采用 ImageNet 预训练的 ResNet18 作为骨干网络，将原 1000 类全连接层替换为 100 类分类头，"
     "在训练集上进行端到端微调。优化器 AdamW（lr=1e-4，weight_decay=1e-4），batch_size=16，训练 12 轮；"
     "每轮在验证集上评估，保存验证准确率最高的模型用于测试。")

heading("3.2 文本单模态基线", 2)
para("采用预训练 BERT (bert-base-uncased) 作为文本编码器，对最后一层隐藏状态按 attention_mask "
     "做掩码平均池化（masked mean pooling）得到句子向量，再接线性层输出 100 类 logits。"
     "优化器 AdamW（lr=2e-5），batch_size=16，训练 10 轮。")

heading("3.3 多模态融合模型", 2)
para("三种融合模型共用相同的编码器：图像侧为 ResNet18（去除 avgpool 与 fc），"
     "文本侧为 BERT；两路特征均通过投影层映射到统一的 256 维空间。"
     "考虑 CPU 训练资源，编码器主干参数冻结，仅训练投影层、融合层与分类头"
     "（可训练参数约 0.5–0.7M，总参数约 121M）。优化器 AdamW（lr=1e-3），batch_size=16，训练 10 轮。")

para("（1）早融合（Early Fusion）：将图像全局特征向量与文本特征向量直接拼接（512 维），"
     "送入两层 MLP 分类器。属于特征级融合，融合发生在决策之前。", bold=False)
para("（2）晚融合（Late Fusion）：图像、文本两路各自经过独立的 MLP 分类头得到 100 维 logits，"
     "将两路 logits 相加作为最终输出。属于决策级融合，两模态独立预测后再合并。")
para("（3）注意力融合（Attention Fusion）：将图像 ResNet 输出的 7×7 特征图展平为 49 个空间 token，"
     "以文本向量作为 query、49 个图像 token 作为 key/value 进行单头跨模态注意力（4 heads），"
     "再经残差连接 + LayerNorm 后接 MLP 分类。该方式让文本特征能够“查询”图像中最相关的空间区域，"
     "属于更细粒度的特征交互。")

# ========== 4 结果 ==========
heading("四、实验结果")
para("五种模型在测试集（913 张）上的整体指标如下表（精确率 / 召回率 / F1 均为宏平均）：")

table = doc.add_table(rows=1, cols=6)
table.style = "Light Grid Accent 1"
table.alignment = WD_TABLE_ALIGNMENT.CENTER
hdr = table.rows[0].cells
for i, t in enumerate(["模型", "best_val_acc", "Accuracy", "Precision", "Recall", "F1"]):
    hdr[i].text = ""
    r = hdr[i].paragraphs[0].add_run(t)
    set_cn(r, "黑体")
    r.bold = True

for name, bv, m in rows:
    cells = table.add_row().cells
    vals = [name,
            f"{bv:.4f}" if bv is not None else "—",
            f"{m['accuracy']:.4f}",
            f"{m['precision_macro']:.4f}",
            f"{m['recall_macro']:.4f}",
            f"{m['f1_macro']:.4f}"]
    for i, v in enumerate(vals):
        cells[i].text = ""
        r = cells[i].paragraphs[0].add_run(v)
        set_cn(r)

doc.add_paragraph()
para("各模型训练过程中的损失与准确率曲线如下图所示：")
doc.add_picture(FIG, width=Cm(15))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
cap = para("图 1  训练损失曲线（左）与准确率曲线（右）")
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER

# ========== 5 分析 ==========
heading("五、结果分析")
para("1. 模态贡献差异显著。图像单模态（Acc=0.791，F1=0.791）大幅领先文本单模态（Acc=0.352，F1=0.328）。"
     "细粒度鸟类分类的关键线索是羽色、喙形、斑纹等视觉细节，而 caption 文本粒度较粗，"
     "不同类别间的文本描述存在大量重叠词汇，100 类区分能力有限。")
para("2. 融合优于文本、但暂低于纯图像。三种融合模型的 Acc 在 0.65–0.68 之间，"
     "较纯文本基线提升约一倍，说明图像特征在融合中起主导作用；"
     "但仍低于图像单模态的 0.791。主要原因是融合实验中两个编码器主干均被冻结，"
     "仅训练轻量投影层，图像特征未能针对本数据集充分适配；同时弱模态（文本）"
     "在特征空间中的噪声会对强模态（图像）产生一定干扰。")
para("3. 三种融合方式对比：早融合（0.684）> 晚融合（0.657）≈ 注意力融合（0.654）。"
     "在冻结特征的设定下，简单拼接 + MLP 已能充分利用两路特征；"
     "注意力机制需要更多可训练参数与数据才能体现优势，在当前冻结主干 + 小数据量的条件下"
     "未能超过早融合。")
para("4. 过拟合情况。图像基线在第 12 轮训练准确率达 0.99，而验证准确率为 0.79，"
     "存在一定过拟合；融合模型由于主干冻结、可训练参数少，训练/验证差距相对较小。")

# ========== 6 结论 ==========
heading("六、结论与展望")
para("本实验完成了图像、文本单模态基线与三种多模态融合模型的搭建与训练，"
     "验证了在细粒度鸟类分类任务中视觉模态的主导作用，以及多模态融合相对纯文本模态的显著提升。"
     "三种融合方式中，早融合以最低的实现复杂度取得了最好的效果。")
para("后续改进方向：（1）解冻编码器主干进行端到端微调（需 GPU 支持），"
     "融合模型性能有望超过图像单模态；（2）文本侧使用全部 10 条 caption 而非随机单条，"
     "或采用 CLIP 等图文对齐的预训练编码器，可增强跨模态语义对齐；"
     "（3）引入更多数据增强与正则化手段缓解图像分支的过拟合。")

# ========== 附录 ==========
heading("附录：代码文件说明")
files = [
    ("00_prepare_data.py", "数据准备：扫描 raw/ 目录，分层切分并生成 annotations_{train,val,test}.json"),
    ("01_singleModal_ImageOnly.py", "图像单模态基线（ResNet18 微调）"),
    ("02_singleModal_TextOnly.py", "文本单模态基线（BERT 微调）"),
    ("03_image_encoder.py", "图像编码器模块（ResNet 系列，支持全局向量/空间 token 两种输出）"),
    ("03_text_encoder.py", "文本编码器模块（BERT，支持 cls/mean 两种池化）"),
    ("04_multimodal_fusion.py", "多模态融合主脚本（--mode early/late/attention/all）"),
]
t2 = doc.add_table(rows=1, cols=2)
t2.style = "Light Grid Accent 1"
for i, t in enumerate(["文件", "说明"]):
    c = t2.rows[0].cells[i]
    c.text = ""
    r = c.paragraphs[0].add_run(t)
    set_cn(r, "黑体")
    r.bold = True
for f, d in files:
    cells = t2.add_row().cells
    cells[0].text = ""
    r0 = cells[0].paragraphs[0].add_run(f)
    r0.font.name = "Consolas"
    set_cn(r0)
    cells[1].text = ""
    r1 = cells[1].paragraphs[0].add_run(d)
    set_cn(r1)

doc.save(OUT)
print("report saved:", OUT)
