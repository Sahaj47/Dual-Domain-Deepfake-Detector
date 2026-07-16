# Explainable Dual-Domain Deepfake Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dual-domain-deepfake-detector.streamlit.app/) This repository contains the code and documentation for an 8-week Summer Term Project conducted at the **Centre for Development of Advanced Computing (C-DAC)**. 

The project focuses on building a lightweight, highly efficient deepfake detection pipeline that analyzes images not just spatially, but in the **frequency domain**, while maintaining mathematical transparency through **Explainable AI (Grad-CAM)**.

---

## 🧠 Project Overview

Modern AI image generators (like Diffusion models and advanced GANs) have become incredibly adept at producing visually flawless images. Traditional spatial-domain detectors often fail to identify these sophisticated forgeries.

This project introduces a **Dual-Domain** approach:
1.  **Frequency Transformation:** We apply Fast Fourier Transform (FFT) to extract hidden, high-frequency mathematical anomalies left behind by generative models.
2.  **Lightweight Deep Learning:** We use a frozen, pre-trained **MobileNetV2** backbone (fine-tuned on CPU) to analyze these spectral fingerprints.
3.  **Explainability:** We integrate **Grad-CAM** to generate heatmaps, proving exactly *where* and *why* the model makes its classification, ensuring trust and transparency.

### Key Achievements
* Successfully balanced and preprocessed a 10,000-image dataset (Real, GAN, Diffusion) under strict local CPU and RAM constraints.
* Achieved an F1-Score of **82.54%** on validation data by targeting high-frequency artifacts.
* Optimized model size down to a remarkably lightweight **11.7 MB**.
* Deployed a zero-footprint, stateless microservice web application using **Streamlit Community Cloud**.

---

## ⚙️ Architecture & Pipeline

The system is designed to be computationally lean, capable of running entirely on standard CPU infrastructure without requiring expensive GPUs.

1.  **Input:** User uploads a `.jpg` or `.png` image.
2.  **Preprocessing:** Image is mathematically transformed using 2D FFT and logarithmically scaled.
3.  **Feature Extraction:** The frequency spectrum is passed through the frozen convolutional layers of MobileNetV2.
4.  **Classification:** A custom, lightweight classification head (3 layers deep) outputs a binary "Real" or "Fake" probability.
5.  **Explainability (Grad-CAM):** The gradients flowing into the final convolutional layer are visualized as a heatmap over the frequency spectrum to highlight anomalous regions.

---

## 🛠️ Repository Structure

```text
├── data/                               # (GitIgnored) Raw spatial images dataset
├── data_fft/                           # (GitIgnored) Extracted FFT representations
├── performance/                        # Validation metrics and charts
│   ├── final_confusion_matrix.png
│   ├── Final_Result.png
│   ├── final_roc_curve.png
│   └── gradcam_result_man_1003.jpg
├── src/                                # Core modeling and training scripts
│   ├── 02_fft_preprocessing.py
│   ├── 04_model_training.py
│   ├── 05_inference.py
│   ├── 06_explainability.py
│   ├── 07_evaluation.py
│   └── model_architecture.py
├── app.py                              # Streamlit UI definition
├── core_engine.py                      # Model loading, FFT math, and Grad-CAM logic
├── deepfake_detector_weights.pth       # The trained 11.7 MB model weights
├── requirements.txt                    # Cloud deployment dependencies
└── README.md                           # Project documentation
