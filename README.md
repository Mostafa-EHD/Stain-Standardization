# Stain-Standardized Deep Learning Framework for Robust Leukocyte Segmentation

![Paper Status](https://img.shields.io/badge/Status-Submitted_to_Information_2026-blue.svg)
![Field](https://img.shields.io/badge/Field-Hematology-red.svg)
![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow/Keras-orange.svg)

Official implementation of the research paper: **"Stain-Standardized Deep Learning Framework for Robust Leukocyte Segmentation Across Heterogeneous Cytological Datasets"**.

---

## 📌 Overview
Accurate leukocyte segmentation is often hindered by staining variability and heterogeneous imaging conditions. This repository provides a **dual-module framework** designed to achieve stain-invariant and robust segmentation by:
1.  **Stain Standardization**: Harmonizing diverse inputs (MGG, H&E, etc.) toward a **Wright-Giemsa** reference appearance.
2.  **Multi-Encoder Segmentation**: Integrating spatial, leukocyte-specific, and nucleus-focused representations from multiple color spaces.



---

## 🏗️ Framework Architecture

### 1. Colorization Module (Stain Normalization)
This module converts inconsistently stained images into a uniform representation.
* **Encoder**: A fine-tuned **VGG16** backbone (up to `block5_conv3`).
* **Bottleneck**: A **Transformer** block with **Multi-Head Self-Attention** (4 heads, key dimension 512) to learn long-range contextual dependencies.
* **Decoder**: A convolutional network that predicts the $(a^*, b^*)$ chrominance channels of the **CIELAB** color space, which are then combined with the original **L (Luminance)** channel.

### 2. Segmentation Module
A multi-branch architecture designed to differentiate biologically distinct elements:
* **RGB Encoder**: Captures global structural context and patterns.
* **WBC Encoder**: Uses **Cyan, Yellow, $b^*$, Hue, Saturation,** and $v^*$ channels to extract leukocyte features.
* **Nucleus Encoder**: Utilizes **Magenta, Luminance, $a^*$, Hue,** and **Saturation** to isolate nuclear characteristics.
* **Refinement**: Includes **Residual Blocks** and integrated thresholding operations (Otsu, Li, Yen, etc.) to enhance robustness against noise.



---

## 📊 Datasets
The framework is evaluated on **six** public and clinical datasets:

| Dataset | Staining | Magnification | Samples | Usage |
| :--- | :--- | :--- | :--- | :--- |
| **AML Cyto. LMU** | Wright-Giemsa | 100x | 18,365 | Reference/Class.  |
| **ALL-IDB2** | MGG | 300x-500x | 260 | Leukemic vs Normal  |
| **LISC** | Wright-Giemsa | 100x | 257 | WBC Classification  |
| **CellaVision** | MGG | 100x | 17,092 | Standardized single cells |
| **JTSC** | H&E | 20x-100x | 300 | Challenging stain conditions |
| **BASE CYTO** | MGG | 100x | 87 | Local clinical dataset  |

---

## 📈 Performance & Metrics
The model is evaluated using the following formal metrics:

* **Accuracy**: $$\frac{TP + TN}{TP + TN + FP + FN}$$ 
* **Dice Coefficient**: $$\frac{2 \times TP}{2 \times TP + FP + FN}$$ 
* **Jaccard Index**: $$\frac{TP}{TP + FP + FN}$$ 

**Key Results**:
* Dice coefficients exceed **96%** on most datasets.
* Average Dice of **97.53%** on AML LMU and **97.48%** on LISC.
* Significant performance gains over baseline U-Net through the synergy of stain normalization and multi-encoder fusion.

---


