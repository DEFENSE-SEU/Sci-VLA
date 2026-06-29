<div align="center">

<p align="center">
  <img src="assets/logo/labvla-symbol.png" width="88" alt="LabVLA symbol" />
  <img src="assets/logo/labvla-wordmark.png" height="56" alt="LabVLA" />
</p>

<h3 align="center"> The First Vision-Language-Action Foundation Model for Scientific Laboratories </h3>

</div>

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&size=24&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&width=1000&lines=Welcome+to+LabVLA;Vision-Language-Action+for+the+Laboratory;Towards+AI+Scientists+That+Can+Physically+Perform+Scientific+Experiments" alt="Typing Animation" />
</div>

<p align="center">
  <img src="assets/paper/first.png" alt="LabVLA Framework" width="92%"/>
</p>

<div align="center">

  <a href='https://github.com/zjunlp/LabVLA'>
    <img src='https://img.shields.io/badge/💻_Code-LabVLA-fc60a8?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e'>
  </a>

  <a href='https://huggingface.co/zjunlp/LabVLA'>
    <img src='https://img.shields.io/badge/🤗_Model-zjunlp%2FLabVLA-ffce44?style=for-the-badge&logoColor=white&labelColor=1a1a2e'>
  </a>

  <a href='https://zjunlp.github.io/LabVLA/'>
    <img src='https://img.shields.io/badge/🔥_Project-Page-00d9ff?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e'>
  </a>

  <a href='https://arxiv.org/abs/2606.13578'>
    <img src='https://img.shields.io/badge/📄_arXiv-2606.13578-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e'>
  </a>

  <a href='https://huggingface.co/papers/2606.13578'>
    <img src='https://img.shields.io/badge/📰_HF_Paper-2606.13578-ffce44?style=for-the-badge&logoColor=white&labelColor=1a1a2e'>
  </a>

  <a href="https://github.com/zjunlp/LabVLA/stargazers">
    <img src='https://img.shields.io/github/stars/zjunlp/LabVLA?color=00d9ff&style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e' />
  </a>

  <a href="https://github.com/zjunlp/LabVLA/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-4ecdc4?style=for-the-badge&logo=open-source-initiative&logoColor=white&labelColor=1a1a2e">
  </a>

</div>

<div align="center">
  <div style="width: 100%; height: 2px; margin: 20px 0; background: linear-gradient(90deg, transparent, #00d9ff, transparent);"></div>
</div>

**LabVLA** turns a **Qwen3-VL-4B-Instruct** vision–language backbone into a real-time robot controller through a **DiT flow-matching action expert**, trained with the π0.5 recipe: FAST action-token pre-training → flow-matching post-training with knowledge insulation → task fine-tuning. This README covers **installation, training, and deployment** — method details are in the [paper](#-citation).

<div align="center">

[✨ Features](#-features) • [📋 TODO](#-todo) • [📦 Installation](#-installation) • [🚀 Quick Start](#-quick-start) • [🎓 Training](#-training) • [🔧 Fine-tuning](#-fine-tuning) • [📡 Deployment](#-deployment) • [📝 Citation](#-citation)

<br />

<img src="assets/media/gifs/02-demo.gif" width="64%" alt="LabVLA Demo" />

</div>

---

## ✨ Features

**🎓 The recipe — every stage in one framework**

| Mode | What it does |
|---|---|
| **VLM pre-training** | FAST action-token cross-entropy on the VLM backbone. |
| **Flow-matching post-training** | Trains the DiT action expert to generate 50-step continuous action chunks. |
| **Knowledge Isolation (KI)** | Stop-gradient between VLM and action expert. |
| **Task fine-tuning** | Fine-tuning for downstream tasks. |
| **Multi-dataset & VQA co-training** | π0-style mixture with homogeneous batches. |
| **delta / abs action modes** | Per-dimension `delta_mask` — arm joints delta, gripper absolute, in one vector. |

**⚙️ Engineering**

- 🚀 **Efficiency** — selective gradient checkpointing (only a subset of modules — e.g. visual encoder or language model — is checkpointed per stage), Liger-Kernel fused ops, DeepSpeed ZeRO-2, and EMA offload together keep per-GPU batch size at **64 on 80 GB A100** with minimal speed penalty.

| Stage | A100 80 GB | BS / GPU | Global BS | ~ s / step |
|---|---|---|---|---|
| VLM Pre-training | 24 (3 × 8) | 64 | 1 536 | ≈ 7 |
| KI Post-training | 16 (2 × 8) | 64 | 1 024 | ≈ 5 |
| Task Fine-tuning | 4 | 48 | 192 | ≈ 3 |

---

## 📋 TODO

- [x] Model weights on Hugging Face
- [x] Inference & deployment code
- [x] **Training & fine-tuning code**
- [ ] **RoboGenesis** & **LabEmbodied-Data** — *coming soon*

The full training, post-training, and fine-tuning code is now available — see [Training](#-training). RoboGenesis and labembodied-data are being organized and will follow soon.

---

## 📦 Installation

**Python 3.10 · CUDA 12.6 · PyTorch 2.7.1** — pinned versions in [`requirements.txt`](requirements.txt).

```bash
conda create -n labvla python=3.10 -y && conda activate labvla

# 1. PyTorch (CUDA 12.6)  →  2. FlashAttention (built against it)  →  3. everything else
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install flash_attn==2.8.3 --no-build-isolation
pip install -r requirements.txt
```

---

## 🚀 Quick Start

**1. Download** the model from Hugging Face:

```bash
huggingface-cli download zjunlp/LabVLA-5B-Base --local-dir LabVLA-5B-Base
```

**2. Deploy** — start a WebSocket inference server:

```bash
PRETRAINED_PATH=LabVLA-5B-Base bash deployment/deploy.sh
```

**3. Evaluate** — connect your robot or simulator client to the server and run rollouts. See [Deployment](#-deployment) for configuration details.

---

## 🎓 Training

One entrypoint (`scripts/train.py`) for all stages, launched through Accelerate + DeepSpeed ZeRO-2 (bf16). Edit the variables at the top of each launch script before running.

### 1 · Prepare data

```bash
python -m data_process scan  --root /path/to/dataset --out /tmp/report.json     # detect bad episodes
python -m data_process clean --src  /path/to/dataset --dst /path/to/clean \
                             --report /tmp/report.json                          # apply report (symlink copy)
python -m data_process stats --dataset /path/to/clean --schema robointer_droid  # normalization stats
```

| Subcommand | Purpose |
|---|---|
| `scan` | Detect bad episodes (corrupt video, decode failures, missing files). |
| `clean` | Apply a scan report as a renumbered symlink copy; originals untouched. |
| `stats` | Compute normalization statistics → `meta/stats.json`. |
| `validate` | Cross-repo integrity checks. |
| `preflight` | Gate a launch on HIGH/CRIT issues before it starts. |

Each dataset is described by an auto-registered `DatasetSchema` ([`src/schema/`](src/schema/)). Add a module under [`schemas/`](schemas/) or pass `--dataset_schema /abs/path.py`.

### 2 · VLM Pre-training

Train the Qwen3-VL backbone on FAST action-token cross-entropy, with robot state discretized into the prompt. Defaults to a joint mixture of `robointer_droid_clean`, `oxe-auge_clean`, `RoboInter-VQA`, and `agibot_world` (π0-style `n^0.43` volume weighting).

```bash
bash launch/vlm_pretrain/train_vlm_pretrain.sh
```

### 3 · Flow-Matching Post-training (Knowledge Insulation)

Train the DiT action expert to generate 50-step continuous action chunks, with stop-gradient between the VLM and the DiT.

```bash
bash launch/ki_posttrain/train_ki_posttrain.sh
```

> **Note:** LabVLA is not limited to the datasets listed above. Extend it to any new dataset by adding a `DatasetSchema` under [`schemas/`](schemas/) — the full pipeline (VLM pre-training, post-training, and fine-tuning) works out of the box.

---

## 🔧 Fine-tuning

Fine-tune a post-trained checkpoint for a downstream task; this does not require the data-preparation step above.

```bash
# Edit launch/finetune/train_labutopia.sh: set PretrainedCkpt, DataRoot,
# RepoIds, DatasetSchema, and ExternalStatsPath to point at your task data.
bash launch/finetune/train_labutopia.sh
```

---

## 📡 Deployment

Download [LabVLA](https://huggingface.co/zjunlp/LabVLA) from Hugging Face, then deploy via the script:

```bash
bash deployment/deploy.sh
```

---

## 📝 Citation

```bibtex
@article{ren2026labvla,
  title   = {LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories},
  author  = {Ren, Baochang and Liu, Xinjie and Chen, Xi and Liu, Yanshuo and
             Li, Chenxi and Gao, Daqi and Su, Zeqin and Xing, Jintao and
             Xue, Zirui and Li, Rui and Zhao, Xiangyu and Qiao, Shuofei and
             Pan, Minting and Zuo, Wangmeng and Bai, Lei and Zhou, Dongzhan and
             Zhang, Ningyu and Chen, Huajun},
  journal = {arXiv preprint arXiv:2606.13578},
  year    = {2026}
}
```

## 🙏 Acknowledgments

Our codebase references [LeRobot](https://github.com/huggingface/lerobot) and [Liger-Kernel](https://github.com/linkedin/Liger-Kernel). We sincerely thank their teams for the outstanding contributions to the open-source community.
