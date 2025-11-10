# Zero-DCE Low-Light Image Enhancement 

This repository contains a PyTorch implementation of the **Zero-Reference Deep Curve Estimation (Zero-DCE)** model for enhancing low-light images.

The goal is to brighten and improve the contrast and color of dark images without introducing noise or artifacts. The key feature of this model is that it's "zero-reference," meaning it was trained **without** needing paired low-light and bright-light images.

[cite_start]The file `zerodce_best_BOLDER.pth` [cite: 1-161] included in this repo is a pre-trained model checkpoint.

---

## 📸 Gallery: Before & After

Here are some representative examples of what Zero-DCE can do.

![Before and After Examples](assets/download.png)
## 🏗️ How Zero-DCE Works: Architecture Overview

The core idea of Zero-DCE is to re-frame low-light enhancement as a task of **deep curve estimation**. Instead of directly manipulating image pixels, the model learns a set of pixel-wise, higher-order curves. These curves are then applied to the input image to adjust its dynamic range.

### The Enhancement Curve

The model estimates a specific enhancement curve $LE(x)$ for each pixel $x$. The enhanced image $R(x)$ is produced by applying this curve to the input image $I(x)$:

$$R(x) = I(x) + \mathcal{A}(x) \times I(x) \times (1 - I(x))$$

* $I(x)$ is the input low-light image's pixel value.
* $R(x)$ is the enhanced output pixel value.
* $\mathcal{A}(x)$ is the **curve parameter map** predicted by the neural network. This map controls the shape of the enhancement curve for every pixel. By iterating this equation multiple times (e.g., 8 times), the model can approximate complex, high-order curves.

### The Network (DCE-Net)

The neural network, called **DCE-Net**, is a lightweight convolutional neural network (CNN).

* **Input:** A low-light image.
* **Output:** The curve parameter map $\mathcal{A}(x)$.
* **Architecture:** It's a simple, encoder-like CNN consisting of several convolutional layers with symmetric concatenation (skip-connections). It does *not* use pooling, which helps maintain the spatial resolution and reduces artifacts.
* **Final Layer:** The network outputs a 24-channel feature map (assuming 8 iterations, as $8 \text{ iterations} \times 3 \text{ RGB channels} = 24$). These are the 24 $\mathcal{A}$ parameters for each pixel.



### Zero-Reference Training

Since there is no "ground truth" bright image to compare against, the model is trained using a set of "non-reference" losses that define what a "good" image looks like:

1.  **Spatial Consistency Loss:** Encourages neighboring pixels to have similar enhancement curves, preserving the natural correlation between adjacent regions.
2.  **Exposure Control Loss:** Pushes the average intensity of the enhanced image towards a predefined "well-exposed" level (e.g., a gray-level value of 0.6).
3.  **Color Constancy Loss:** Prevents color shifting by penalizing any drift in the relative balance of the R, G, and B channels.
4.  **Illumination Smoothness Loss:** Keeps the predicted curve parameter map $\mathcal{A}(x)$ smooth, preventing abrupt changes and noise.

---

## 🚀 How to Use This Repo

### 1. Requirements

You'll need Python and the following libraries:

```bash
pip install torch torchvision numpy opencv-python pillow

```







