# Stain-Standardized Deep Learning Framework for Robust Leukocyte Segmentation

![Paper Status](https://img.shields.io/badge/Status-Submitted%20(Information%202026)-2f6fdb.svg)
![Task](https://img.shields.io/badge/Task-Leukocyte%20Segmentation-8b0000.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-f39c12.svg)

Official implementation of the research paper:

**"Stain-Standardized Deep Learning Framework for Robust Leukocyte Segmentation Across Heterogeneous Cytological Datasets"**

---

## 📌 Important Training Protocol

Both the **colorization (stain standardization)** and **segmentation** modules are trained **exclusively on the AML-Cytomorphology LMU dataset (Wright–Giemsa staining)**.

The remaining datasets (**LISC, CellaVision, JTSC, ALL-IDB2, BASE CYTO**) are used **strictly for external evaluation**, with:

- No retraining  
- No fine-tuning  
- No dataset-specific adaptation  

This design allows direct assessment of **cross-dataset generalization** and **robustness to staining variability**.

---

## 📖 Overview

Accurate leukocyte segmentation is frequently degraded by:

- Staining variability (Wright–Giemsa, MGG, H&E)
- Differences in magnification and acquisition devices
- Heterogeneous cell density and background noise

This repository implements a **two-stage deep learning pipeline**:

1. **Stain Standardization Module**  
   Harmonizes input images toward a Wright–Giemsa reference appearance.

2. **Multi-Encoder Segmentation Module**  
   Extracts complementary representations across multiple color spaces for robust segmentation.

The goal is simple:

> Train once. Generalize everywhere.

---

## 🏗️ Methodology

### 1️⃣ Colorization Module (Stain Standardization)

This module converts heterogeneous stain appearances into a unified Wright–Giemsa-like representation.

**Architecture:**

- **Encoder:** Fine-tuned VGG16 (up to `block5_conv3`)
- **Bottleneck:** Transformer block with Multi-Head Self-Attention  
  - 4 attention heads  
  - Key dimension: 512
- **Decoder:** Convolutional decoder predicting the **CIELAB chrominance channels** \((a^\*, b^\*)\)

The predicted chrominance channels are combined with the original **L (luminance)** channel to reconstruct the standardized image.

---

### 2️⃣ Multi-Encoder Segmentation Module

A multi-branch architecture designed to capture biologically meaningful features.

#### 🔹 RGB Encoder
Captures global morphology, spatial structure, and texture.

#### 🔹 Leukocyte-Focused Encoder
Uses discriminative channels such as:
- Cyan
- Yellow
- \(b^\*\)
- Hue
- Saturation
- \(v^\*\)

#### 🔹 Nucleus-Focused Encoder
Emphasizes nuclear structures using:
- Magenta
- Luminance
- \(a^\*\)
- Hue
- Saturation

#### 🔹 Refinement
Includes:
- Residual blocks
- Optional threshold-based priors (Otsu, Li, Yen, etc.)

These components increase robustness to noise and domain shift.

---

## 📊 Datasets

| Dataset | Staining | Magnification | Samples | Role |
|----------|------------|----------------|----------|-------|
| **AML-Cytomorphology LMU** | Wright–Giemsa | 100× | 18,365 | Training / Validation / Test |
| **LISC** | Wright–Giemsa | 100× | 257 | External evaluation |
| **CellaVision** | MGG | 100× | 17,092 | External evaluation |
| **JTSC** | H&E | 20×–100× | 300 | External evaluation |
| **ALL-IDB2** | MGG | 300×–500× | 260 | External evaluation |
| **BASE CYTO** | MGG | 100× | 87 | External evaluation |

Evaluation on the five non-AML datasets is performed using the trained AML model without adaptation.

---

## 📈 Evaluation Metrics

- **Dice Coefficient**  
  \[
  \frac{2TP}{2TP + FP + FN}
  \]

- **Jaccard Index (IoU)**  
  \[
  \frac{TP}{TP + FP + FN}
  \]

- **Accuracy**  
  \[
  \frac{TP + TN}{TP + TN + FP + FN}
  \]

---

## 📈 Performance Highlights (Dice)

- AML LMU: **97.53%**
- LISC: **97.48%**
- CellaVision: **97.47%**
- JTSC: **97.90%**
- ALL-IDB2: **98.73%**
- BASE CYTO: **96.09%**

These results demonstrate strong cross-dataset generalization from a single training dataset.

---

## 📦 Pretrained Weights

### 🎨 Colorization Module

The following pretrained weights are provided for the stain standardization module:

- `colorisation_vgg16_encoder.h5`
- `colorisation_decoder.h5`

---

### 🧠 Segmentation Module

Pretrained colorization and segmentation weights can be downloaded from:

🔗 https://drive.google.com/drive/folders/1eImjTdsLICvj7OEeZHwxyEicbr6_erbH?usp=sharing
