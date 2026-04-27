# WF-Diff Extensions: Improving Underwater Image Restoration Metrics

## Project Metadata
### Authors
- **Team:** Abdulaziz Alfaraj, Hassan Al Nasser, Mohammed Al Naser
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** KFUPM

---

## Introduction
Underwater images often suffer from **color cast**, **low contrast**, **blur**, and **loss of fine details** due to light absorption and scattering in water. These degradations reduce both visual quality and downstream performance for underwater applications such as inspection, robotics, and tracking.

WF-Diff is a strong Underwater Image Enhancement/Restoration (UIE/UIR) framework that combines:
1. **Frequency-domain processing** (Wavelet + Fourier interaction) for preliminary enhancement.
2. **Diffusion-based refinement** in frequency space to recover high-frequency details and improve realism.

In this project, we build on top of WF-Diff's public implementation and contribute **WaveFlow-UIE**, a wavelet-domain flow-based extension aimed at improving restoration quality and runtime under a consistent benchmark protocol.

**Introductory image:**
![WF-Diff Overview](assets/intro.png)

---

## Problem Statement
We treat underwater image restoration as an **image-to-image enhancement** task:

- **Input:** degraded underwater image
- **Output:** enhanced/restored image with reduced color distortion and improved texture/detail

Because the field can be sensitive to evaluation settings (for example, native resolution vs resized evaluation), we report results under clearly stated protocols and focus on improvements measurable by standard restoration metrics.

**Evaluation metrics**
- **PSNR / SSIM** for paired fidelity
- **LPIPS / FID** for perceptual quality
- **UIQM / UCIQE** for non-reference underwater quality

**Key Questions**
- **Q1:** Which changes contribute most to improving restoration quality?
- **Q2:** Do improvements generalize across datasets and underwater conditions?
- **Q3:** What is the tradeoff between visual quality and computational cost?

---

## Application Area and Project Domain
Underwater image restoration is important for:
- Underwater robotics and navigation
- Marine inspection and infrastructure monitoring
- Underwater object detection and tracking pipelines
- Scientific imaging and exploration

---

## What is the paper trying to do, and what are you planning to do?
### What the reference paper does (WF-Diff)
WF-Diff proposes a two-stage framework:
1. **WFI2-net (Wavelet-based Fourier Information Interaction Network)**
   - preliminary enhancement in the **wavelet space**
2. **FRDAM (Frequency Residual Diffusion Adjustment Module)**
   - diffusion-based refinement of low- and high-frequency residuals

### What we are planning to do
We build on the WF-Diff framework and contribute **research-level improvements** aimed at increasing restoration quality metrics and visual fidelity. The repository includes:
- model-level and training-level extensions
- reproducible experiments and ablations
- quantitative results on multiple datasets
- qualitative comparisons and runtime analysis

---

## Model Description

This repository no longer only mirrors WF-Diff. It contains **WaveFlow-UIE**, our main model variant.

WaveFlow-UIE is a **wavelet-domain rectified flow** model for underwater image enhancement. The core implementation is in:

- [waveflow_uie/models/waveflow.py](waveflow_uie/models/waveflow.py)
- [waveflow_uie/models/physics_prior.py](waveflow_uie/models/physics_prior.py)
- [waveflow_uie/models/velocity_unet.py](waveflow_uie/models/velocity_unet.py)
- [waveflow_uie/losses.py](waveflow_uie/losses.py)

### High-level pipeline
1. **Wavelet decomposition**
   - degraded RGB input is mapped to a 12-channel Haar wavelet representation
2. **Physics prior conditioning**
   - a lightweight branch estimates:
     - transmission map `t(x)`
     - ambient light `A`
3. **Velocity prediction**
   - a `VelocityUNet` predicts enhancement velocity directly in wavelet space
4. **Small-step ODE integration**
   - the wavelet state is updated for a small number of flow steps
5. **Inverse wavelet reconstruction**
   - enhanced wavelets are mapped back to RGB

### Training objective
WaveFlow-UIE combines:
- flow matching loss
- frequency-weighted wavelet loss
- LPIPS perceptual loss
- Lab color consistency loss
- optional physics auxiliary loss

### WaveFlow checkpoints used in this project
- **WaveFlow UIEB**
  - specialist checkpoint for `UIEB`
- **WaveFlow LSUI**
  - specialist checkpoint for `LSUI`
  - also used on `UFO-120` and `EUVP`
- **WaveFlow BOTH**
  - mixed-data checkpoint for `UIEB + LSUI`
  - mainly used on `U45` and `C60`

---

## Project Documents
- **Presentation PDF:** [Project Presentation](./CVProjectPresentation1.pdf)
- **Presentation PPTX:** [Project Presentation](./CVProjectPresentation1.pptx)
- **Term Paper PDF:** [Term Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Wavelet-based_Fourier_Information_Interaction_with_Frequency_Diffusion_Adjustment_for_Underwater_CVPR_2024_paper.pdf)
- **Term Paper Latex Files:** [Term Paper Latex files](https://arxiv.org/src/2311.16845)

---

## References

### Reference Paper
- **WF-Diff (CVPR 2024):** Wavelet-based Fourier Information Interaction with Frequency Diffusion Adjustment for Underwater Image Restoration
  - CVF OpenAccess: https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Wavelet-based_Fourier_Information_Interaction_with_Frequency_Diffusion_Adjustment_for_Underwater_CVPR_2024_paper.html
  - PDF: https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Wavelet-based_Fourier_Information_Interaction_with_Frequency_Diffusion_Adjustment_for_Underwater_CVPR_2024_paper.pdf

### Reference GitHub (Upstream Implementation)
- **WF-Diff official repo:** https://github.com/ChenzhaoNju/WF-Diff

### Reference Dataset
- **UIEB (upstream split link):** https://pan.baidu.com/s/1BWtIPz-xUDaatsncOFCJHg?pwd=123x
- **LSUI (upstream split link):** https://pan.baidu.com/s/1-Nk8iqmOVIl3ulZTHkdpbQ?pwd=123x
- **Extraction code:** 123x

---

## Project Technicalities

### Project UI
- Primary usage is via **command-line training/testing scripts**
- The project also includes a **benchmark harness** for public baselines
- Outputs include:
  - CSV benchmark summaries
  - LaTeX report tables
  - qualitative comparison images for slides

### Terminologies
- **UIE/UIR:** Underwater Image Enhancement/Restoration
- **Wavelet Transform (DWT/IWT):** Decomposes an image into low/high-frequency subbands
- **Frequency Domain:** Low-frequency carries structure/color, high-frequency carries edges/details
- **Rectified Flow / Conditional Flow Matching:** Flow-based training where the model predicts enhancement velocity directly
- **Physics Prior Branch:** Conditioning branch that predicts transmission and ambient-light cues
- **PSNR / SSIM:** Standard paired restoration metrics
- **LPIPS / FID:** Perceptual quality metrics
- **UCIQE / UIQM:** Non-reference underwater quality metrics

### Problem Statements
- **Problem 1:** Underwater degradations are diverse and hard to model with pixel-only methods
- **Problem 2:** Restoration quality depends heavily on frequency details and cross-frequency interactions
- **Problem 3:** Diffusion-style refinement improves quality but increases runtime
- **Problem 4:** Public benchmark splits and released checkpoints are inconsistent across papers

### Datasets and benchmark setup actually used
This repository uses a **unified reproduced benchmark** on the public split family available locally.

Current dataset set:
- **UIEB:** `800 train / 90 test`
- **LSUI:** `3879 train / 400 test`
- **UFO-120:** `120` paired test images
- **EUVP test samples:** `515` paired test pairs
- **U45:** non-reference
- **C60:** non-reference

Grouped as:
- **Paired / full-reference:** `UIEB`, `LSUI`, `UFO-120`, `EUVP`
- **Non-reference:** `U45`, `C60`

### Benchmarked models currently integrated
- `WaveFlow-UIE`
- `WF-Diff`
- `DiffWater / DM-water`
- `SCNet`
- `UIE-WD`
- `U-shape`
- `UIEC^2-Net`
- `Water-Net`
- `UDAformer`
- `UWCNN`

### Current benchmark claim supported by the codebase
The strongest defensible reading of this repository is:

> Under a unified reproduced benchmark on public splits and public checkpoints, WaveFlow-UIE improves over the released WF-Diff baseline in quality-speed tradeoff and remains competitive with strong feedforward underwater enhancement baselines.

---

## Metric Tables

Use this section as the placeholder for the final report tables.

### Table A - UIEB and LSUI at 256x256
- source: [results/report/table_a_uieb_lsui_256.tex](results/report/table_a_uieb_lsui_256.tex)

### Table B - UIEB and LSUI at native resolution
- source: [results/report/table_a_uieb_lsui_native.tex](results/report/table_a_uieb_lsui_native.tex)

### Table C - UFO-120, EUVP, U45, and C60 at 256x256
- source: [results/report/table_b_other_datasets_256.tex](results/report/table_b_other_datasets_256.tex)

### Table D - Inference time
- source: [results/report/table_runtime_uieb_lsui.tex](results/report/table_runtime_uieb_lsui.tex)

---

## Model Workflow
WF-Diff-style restoration follows this high-level pipeline:

1. **Input:** underwater image
2. **Wavelet decomposition (DWT):** split image into low/high-frequency subbands
3. **Frequency preliminary enhancement (WFI2-net):** enhance frequency representations and cross-frequency interaction
4. **Initial enhanced image:** reconstruct coarse enhancement
5. **Frequency diffusion adjustment (FRDAM):** refine low/high-frequency residuals using diffusion-based adjustment
6. **Output:** final restored underwater image

---

## How to Run the Code

### 1) Clone the Repository
```bash
git clone https://github.com/BRAIN-Lab-AI/WaveLift-Fast-Frequency-Guided-Underwater-Restoration.git
cd WaveLift-Fast-Frequency-Guided-Underwater-Restoration
```

### 2) Set Up the Environment
```bash
conda create -n wfdiff_ext python=3.10 -y
conda activate wfdiff_ext
pip install -r requirements.txt
```

### 3) Download Datasets
Download the upstream UIEB and LSUI links if needed, then place the dataset folders locally and update the paths in the YAML configs.

### 4) Train baseline WF-Diff
```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_Wfdiff.yml
```

### 5) Test baseline WF-Diff
```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/test_wfdiff.yml
```

---

## How to Train the Model

This repository supports two training paths:
1. **Baseline WF-Diff** using the upstream BasicSR-style trainer
2. **WaveFlow-UIE** using our flow-based trainer

### 1) Prepare the dataset
Organize paired underwater images as:
```text
data/
└── UIEB/
    ├── train/
    │   ├── input/
    │   └── target/
    └── test/
        ├── input/
        └── target/
```

### 2) Pick a config
**Baseline WF-Diff:** [options/train/train_Wfdiff.yml](options/train/train_Wfdiff.yml)

**WaveFlow-UIE:** choose one from [configs/](configs/):
- `waveflow_uie_uieb.yaml`
- `waveflow_uie_no_physics.yaml`
- `waveflow_uie_no_freq_weight.yaml`
- `waveflow_uie_equal_loss.yaml`

### 3) Launch training
**Baseline WF-Diff**
```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_Wfdiff.yml
```

**WaveFlow-UIE**
```bash
python -m waveflow_uie.train --config configs/waveflow_uie_uieb.yaml --seed 42
```

**Resume**
```bash
python -m waveflow_uie.train --config configs/waveflow_uie_uieb.yaml --resume experiments/<run_name>/checkpoints/latest.pt
```

### 4) Monitor progress
- Console logs
- TensorBoard
- validation metrics and saved checkpoints

### 5) Training notes
- FP16 is enabled by default
- effective batch size = `batch_size * grad_accum_steps`
- keep the seed fixed for ablations
