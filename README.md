# Stain-Standardized Deep Learning Framework for Robust Leukocyte Segmentation

![Paper Status](https://img.shields.io/badge/Status-Submitted_to_Information_2026-blue.svg)
![Field](https://img.shields.io/badge/Field-Hematology-red.svg)
![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow/Keras-orange.svg)

Official implementation of the research paper: **"Stain-Standardized Deep Learning Framework for Robust Leukocyte Segmentation Across Heterogeneous Cytological Datasets"**.

---

## 📌 Overview
[cite_start]Accurate leukocyte segmentation is often hindered by staining variability and heterogeneous imaging conditions[cite: 14]. [cite_start]This repository provides a **dual-module framework** designed to achieve stain-invariant and robust segmentation by[cite: 15]:
1.  [cite_start]**Stain Standardization**: Harmonizing diverse inputs (MGG, H&E, etc.) toward a **Wright-Giemsa** reference appearance[cite: 16, 249].
2.  [cite_start]**Multi-Encoder Segmentation**: Integrating spatial, leukocyte-specific, and nucleus-focused representations from multiple color spaces[cite: 17].



---

## 🏗️ Framework Architecture

### 1. Colorization Module (Stain Normalization)
[cite_start]This module converts inconsistently stained images into a uniform representation[cite: 171, 172].
* [cite_start]**Encoder**: A fine-tuned **VGG16** backbone (up to `block5_conv3`)[cite: 511].
* [cite_start]**Bottleneck**: A **Transformer** block with **Multi-Head Self-Attention** (4 heads, key dimension 512) to learn long-range contextual dependencies[cite: 518, 519].
* [cite_start]**Decoder**: A convolutional network that predicts the $(a^*, b^*)$ chrominance channels of the **CIELAB** color space, which are then combined with the original **L (Luminance)** channel[cite: 564, 565].

### 2. Segmentation Module
[cite_start]A multi-branch architecture designed to differentiate biologically distinct elements[cite: 584, 672]:
* [cite_start]**RGB Encoder**: Captures global structural context and patterns[cite: 587, 588].
* [cite_start]**WBC Encoder**: Uses **Cyan, Yellow, $b^*$, Hue, Saturation,** and $v^*$ channels to extract leukocyte features[cite: 598].
* [cite_start]**Nucleus Encoder**: Utilizes **Magenta, Luminance, $a^*$, Hue,** and **Saturation** to isolate nuclear characteristics[cite: 658].
* **Refinement**: Includes **Residual Blocks** and integrated thresholding operations (Otsu, Li, Yen, etc.) to enhance robustness against noise[cite: 593, 595, 601, 775].



---

## 📊 Datasets
The framework is evaluated on **six** public and clinical datasets[cite: 18]:

| Dataset | Staining | Magnification | Samples | Usage |
| :--- | :--- | :--- | :--- | :--- |
| **AML Cyto. LMU** | Wright-Giemsa | 100x | 18,365 | Reference/Class. [cite: 244] |
| **ALL-IDB2** | MGG | 300x-500x | 260 | Leukemic vs Normal [cite: 244, 255] |
| **LISC** | Wright-Giemsa | 100x | 257 | WBC Classification [cite: 244, 256] |
| **CellaVision** | MGG | 100x | 17,092 | Standardized single cells [cite: 244, 282] |
| **JTSC** | H&E | 20x-100x | 300 | Challenging stain conditions [cite: 244, 284] |
| **BASE CYTO** | MGG | 100x | 87 | Local clinical dataset [cite: 244, 288] |

---

## 📈 Performance & Metrics
The model is evaluated using the following formal metrics[cite: 790]:

* [cite_start]**Accuracy**: $$\frac{TP + TN}{TP + TN + FP + FN}$$ [cite: 793, 794]
* [cite_start]**Dice Coefficient**: $$\frac{2 \times TP}{2 \times TP + FP + FN}$$ [cite: 798, 804]
* **Jaccard Index**: $$\frac{TP}{TP + FP + FN}$$ [cite: 797, 802]

**Key Results**:
* [cite_start]Dice coefficients exceed **96%** on most datasets[cite: 19].
* [cite_start]Average Dice of **97.53%** on AML LMU and **97.48%** on LISC[cite: 843, 847].
* Significant performance gains over baseline U-Net through the synergy of stain normalization and multi-encoder fusion[cite: 971].

---

## 📜 Citation
If you find this work useful for your research, please cite our paper:

```bibtex
@article{lazouni2026stain,
  title={Stain-Standardized Deep Learning Framework for Robust Leukocyte Segmentation Across Heterogeneous Cytological Datasets},
  author={Lazouni, L.R. and Benazzouz, M. and Hadjila, F. and Lazouni, M.E.A. and El Habib Daho, M.},
  journal={Information},
  year={2026},
  publisher={MDPI}
}
