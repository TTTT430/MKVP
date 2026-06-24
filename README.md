# MKVP: Multi-source Knowledge guided Visual Confidence Perception for Multimodal Sentiment Analysis

> **论文标题**：基于多源知识引导的视觉置信度感知的多模态情感分析网络  
> **发表期刊**：《电子与信息学报》(Journal of Electronics & Information Technology)

## 概述

MKVP（基于多源知识引导的视觉置信度感知网络）是一个面向多模态情感分析任务的深度学习模型。针对图文情感分析中普遍存在的**视觉环境噪声**、**图文情感不一致**以及**模态贡献不平衡**三大问题，MKVP 创新性地提出了以下核心模块：

1. **多源知识引导矩阵 (Multi-source Knowledge Guidance Matrix)**：融合句法依存 (spaCy)、情感极性 (SenticNet)、方面词共现三类语言学知识，构建语义增强矩阵；
2. **视觉置信度感知模块 (Visual Confidence Perception, VCP)**：基于多源知识引导矩阵驱动，动态量化视觉特征与文本语义的亲和度，抑制无关视觉噪声；
3. **双流并行交互模块 (Dual-stream Parallel Interaction)**：通过四路 Transformer（T→T, I→I, I→T, T→I）实现图文特征深度对齐；
4. **全局门控融合机制 (Global Gated Fusion)**：动态调整单模态与跨模态特征的融合权重；
5. **联合损失函数**：结合标签平滑交叉熵损失、双向 KL 散度一致性约束与 FGM 对抗训练。

## 模型架构

### 图 1：MKVP 整体模型架构

![MKVP 模型架构图](模型图.png)

模型由文本编码器 (BERT)、图像编码器 (ResNet)、多源知识引导矩阵、视觉置信度感知 (VCP) 模块、双流并行交互模块以及全局门控融合机制组成。

### 图 2：多源知识引导矩阵

![多源知识引导矩阵](多源知识引导矩阵.png)

多源知识引导矩阵融合了三类语言学知识：① 句法依存矩阵（spaCy 依存解析，捕捉词语间的结构关系）；② 情感极性矩阵（SenticNet 情感词典，编码词语的情感强度）；③ 方面词共现矩阵（增强与方面词相关的语义关联）。三类矩阵逐元素融合后经 LayerNorm 归一化，得到最终的多源知识引导矩阵 **M**。

### 图 3：视觉置信度感知 (VCP) 模块

![VCP模块](VCP模块.png)

VCP 模块利用多源知识引导矩阵 **M** 生成查询向量，与图像特征 **F_I** 的键向量进行交叉注意力计算，得到视觉置信度矩阵 **A_per**。通过置信度加权聚合，输出高置信度的图像特征 **F_per**，有效抑制视觉噪声。

## 主要实验结果

| 模型 | MVSA-Single (Acc/F1) | MVSA-Multiple (Acc/F1) | HFM (Acc/F1) |
|------|---------------------|------------------------|--------------|
| CNN | 68.19 / 55.90 | 65.64 / 57.66 | 80.03 / 75.72 |
| Bi-LSTM | 70.12 / 65.06 | 67.90 / 67.90 | 81.90 / 77.53 |
| BERT | 71.11 / 69.70 | 67.59 / 66.24 | 83.39 / 83.26 |
| MGNNS | 73.77 / 72.70 | 72.49 / 69.34 | — |
| CLMLF | 75.11 / 73.02 | 70.53 / 68.45 | 81.74 / 78.74 |
| GIGNN | 75.11 / 73.33 | 73.41 / 70.96 | 84.02 / 80.60 |
| DIB | 76.05 / 75.20 | — | — |
| MVCN | 76.06 / 74.55 | 72.07 / 70.01 | 85.43 / 84.87 |
| MFGFN | 76.22 / 75.38 | 70.82 / 69.94 | 85.56 / 84.87 |
| D2R | 76.67 / 75.59 | 71.59 / 70.85 | 85.68 / 85.23 |
| DTN | 77.11 / 76.46 | 72.72 / 72.72 | — |
| MIGSIE | 76.40 / 75.20 | 70.70 / 68.10 | — |
| **MKVP (Ours)** | **77.56 / 76.69** | **72.72 / 70.66** | **87.26 / 86.78** |

### 模型复杂度

| 指标 | 数值 |
|------|------|
| 参数量 (Params) | 175.11M |
| 计算量 (FLOPs) | 22.48G |
| 推理时间 (单样本) | 13.66ms |

## 目录结构

```
MKVP/
├── main.py                 # 主程序入口，参数解析，训练/测试流程
├── model.py                # 核心模型定义 (Transformer, VCP, 门控融合, CLModel)
├── pre_model.py            # 自定义 RoBERTa Encoder 实现
├── train_process.py        # 训练流程 (含 EMA + FGM 对抗训练)
├── dev_process.py          # 验证集评估
├── test_process.py         # 测试集评估
├── data_process.py         # 数据加载、文本/图像增强、批处理
├── preprocess_ate.py       # 方面词 (Aspect Words) 离线提取脚本
├── checkpoint/             # 预训练模型与训练保存目录
├── dataset/                # 实验数据集 (HFM / MVSA)
├── util/
│   ├── compare_to_save.py  # 模型择优保存工具
│   ├── write_file.py       # 日志写入工具
│   ├── text_process_fun.py # 文本处理函数
│   └── image_augmentation/ # 图像增强 (RandAugment)
└── 电子与信息学报终版修订.pdf # 已发表论文终稿
```

## 环境依赖

- **Python** >= 3.8
- **PyTorch** >= 1.10
- **Transformers** (HuggingFace) >= 4.20
- **torchvision** >= 0.11
- **spaCy** >= 3.0 (`en_core_web_sm`)
- **SenticNet** (自建情感词典模块)
- **scikit-learn**
- **Pillow**
- **tqdm**
- **matplotlib**
- **tensorboard**

### 安装依赖

```bash
pip install torch torchvision transformers spacy scikit-learn pillow tqdm matplotlib tensorboard
python -m spacy download en_core_web_sm
```

## 数据集准备

项目支持三个多模态情感分析数据集：

| 数据集 | 规模 (Train/Val/Test) | 来源 |
|--------|----------------------|------|
| MVSA-Single | 3611 / 450 / 450 | Twitter 图文对 |
| MVSA-Multiple | 13624 / 1700 / 1700 | Twitter 图文对 |
| HFM | 19816 / 2410 / 2409 | 社交媒体多模态 |

### 数据预处理

1. 使用 `preprocess_ate.py` 离线提取方面词 (aspect words)：

```bash
python preprocess_ate.py
```

该脚本会为各数据集的 `train.json` / `dev.json` / `test.json` 生成包含 `aspect_words` 字段的 `*_processed.json` 文件。

2. 确保 `dataset/data/` 目录结构如下：

```
dataset/data/
├── HFM/
│   ├── train_processed.json
│   ├── valid_processed.json
│   ├── test_processed.json
│   ├── dataset_image/        # 图片文件
│   └── HFM.json              # 翻译数据
├── MVSA-single/
│   ├── 10-flod-1/
│   │   ├── train_processed.json
│   │   ├── dev_processed.json
│   │   ├── test_processed.json
│   └── dataset_image/
└── MVSA-multiple/
    ├── 10-flod-1/
    │   ├── train_processed.json
    │   ├── dev_processed.json
    │   ├── test_processed.json
    └── dataset_image/
```

## 预训练模型

放置于 `checkpoint/` 目录下：

```bash
checkpoint/
├── txt/
│   ├── bert-base-uncased/    # BERT 模型与词表
│   └── bertweet-base/        # BERTweet 模型
└── best_model/               # 保存的最优模型
    └── best-model.pth
```

## 使用方法

### 训练

```bash
python main.py \
    -run_type 1 \
    -data_type HFM \
    -text_model bert-base \
    -image_model resnet-152 \
    -batch_size 32 \
    -epoch 40 \
    -lr 5e-5 \
    -fuse_lr 5e-5 \
    -gpu_num 0 \
    -lambda_ce 0.1 \
    -lambda_kl 1.0 \
    -lambda_adv 1.0 \
    -save_model_path checkpoint \
    -add_note "experiment_1"
```

### 测试

```bash
python main.py \
    -run_type 2 \
    -data_type HFM \
    -text_model bert-base \
    -image_model resnet-152 \
    -gpu_num 0
```

### 关键超参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `-run_type` | 1: 训练, 2: 测试 | 1 |
| `-data_type` | 数据集: HFM / MVSA-single / MVSA-multiple | HFM |
| `-text_model` | 文本编码器: bert-base / bertweet | bert-base |
| `-image_model` | 图像编码器: resnet-{18,34,50,101,152} | resnet-152 |
| `-batch_size` | 批次大小 | 32 (HFM), 16 (MVSA-Multiple) |
| `-lr` | 学习率 | 5e-5 (HFM), 2e-5 (MVSA-Multiple) |
| `-lambda_ce` | 交叉熵损失权重 | 0.1 |
| `-lambda_kl` | KL 散度一致性损失权重 | 1.0 |
| `-lambda_adv` | 对抗样本损失权重 | 1.0 |
| `-l_dropout` | Dropout 比率 | 0.3 (HFM), 0.5 (MVSA-Multiple) |
| `-fuse_type` | 融合方式: ave / max / att | ave |
| `-optim` | 优化器: adam / adamw / sgd | adamw |
| `-activate_fun` | 激活函数: relu / gelu | gelu |
| `-image_output_type` | 图像输出: all (全局+区域) / cls | all |
| `-tran_dim` | Transformer 输入维度 | 768 |
| `-train_fuse_model_epoch` | 仅训练融合层的预热 epoch 数 | 0 |
| `-fixed_image_model` | 是否冻结图像模型参数 | False |

## 训练策略

### 联合损失函数

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{cls} + \lambda_2 \mathcal{L}_{cl} + \lambda_3 \mathcal{L}_{adv}$$

- **$\mathcal{L}_{cls}$**：标签平滑交叉熵损失（三路输出：纯文本、纯图像、多模态融合）
- **$\mathcal{L}_{cl}$**：双向 KL 散度一致性约束（原始样本 ↔ 增强样本 → 对比学习）
- **$\mathcal{L}_{adv}$**：FGM (Fast Gradient Method) 对抗扰动损失（在词嵌入层施加扰动）

### 其他训练技巧

- **EMA (Exponential Moving Average)**：权重指数移动平均，平滑模型参数（衰减率 0.999）
- **梯度累积**：通过 `-acc_grad` 控制，等效增大批次大小
- **学习率调度**：线性 warmup + 线性衰减
- **数据增强**：
  - 文本：随机删除 (delete) / 随机乱序 (shuffle)，噪声比 0%~50%
  - 图像：RandAugment (N=2, M=14)

## 开源数据

项目数据可在科学数据银行获取：[https://www.scidb.cn/s/Fji67r](https://www.scidb.cn/s/Fji67r)

## 许可证

本项目仅限学术研究使用。商用请联系作者。

---

*如有问题请联系：29859491@qq.com*
