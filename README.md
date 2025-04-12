# DeepBreath AI 🫁 – Lung Cancer Malignancy Detection

> **Empowering early lung cancer detection with advanced deep learning models.** 🚀

DeepBreath AI is an end-to-end solution designed to assist radiologists and healthcare professionals in **identifying and classifying lung cancer malignancies** from CT scans. By leveraging advanced machine learning and data preprocessing techniques, our goal is to **accelerate diagnosis** and **improve patient outcomes**.

---

## Table of Contents
1. [About the Project](#about-the-project-)
2. [Dataset](#dataset-)
3. [Methods](#methods-)
4. [Results](#results-)
5. [Conclusion](#conclusion-)
6. [Known Issues](#known-issues-)
7. [Hardware Requirements](#hardware-requirements-)
8. [Installation & Usage](#installation--usage-)
9. [References](#references-)

---

## About the Project 🏥
**Lung cancer (LC)** is one of the leading causes of cancer-related deaths worldwide. **Early detection** is crucial for improving survival rates. However, manual analysis of **CT scans** can be time-consuming and prone to errors.

### What Does DeepBreath AI Do? 💡
1. **Binary Malignancy Classification** 🔵  
   - Distinguishes **benign** (Classes 1, 2, 3) from **malignant** (Classes 4, 5) tumors.
2. **Multi-class Malignancy Classification** 🌈  
   - Predicts one of five malignancy scores (1–5).
3. **Full-Slice vs. Zoomed-Slice Analysis** 🔍  
   - Evaluates performance on both the **entire CT slice** and a **zoomed-in nodule slice** to better capture tumor details.

### Authors 🧑‍💻
- **Sofia Brunori**  
- **Alberto Eusebio**  
- **Alessandro Tognotti**  
- **Andrea Tomasella**

---

## Dataset 📦
Our dataset includes **CT scan slices** from **2,363 patients**, provided in **NRRD format** with Hounsfield Unit (HU) values.

- Each patient has:
  1. A **full CT slice** (512×512 int16 matrix)
  2. A **zoomed-in slice** focusing on the nodule

### Class Distribution 📊
- **Binary**: 1,793 benign vs. 570 malignant  
- **Multi-class**: 5 imbalance-prone classes (1–5 malignancy scores)

##### Data avilability🚨: The data is not made publicly available, as consense from the authors of the dataset was negated.
---

## Methods 🧪
### Data Preprocessing 🔧
1. **Normalization**  
   - Min-max normalization to scale values between [0, 1].
2. **HU Windowing Transformation**  
   - Emphasizes lung tumors using a window width of **1300** and a window level of **-220**.
3. **Augmentation**  
   - Applied for both binary and multi-class tasks to improve generalization.
4. **Balancing Techniques**  
   - **Binary**: Random oversampling  
   - **Multi-class**: SMOTE + class weight balancing

### Model Architectures 🔮
We utilize **EfficientNet** variants, each customized for the specific task:

| Task                          | Architecture       | Notes                                        |
|-------------------------------|--------------------|----------------------------------------------|
| **Full-slice Binary**         | EfficientNetV2S   | Fine-tuning last 10 layers                  |
| **Full-slice Multi-class**    | EfficientNetB1    | Class balancing + advanced preprocessing     |
| **Zoomed-slice Binary**       | EfficientNetV2S   | SMOTE augmentation                           |
| **Zoomed-slice Multi-class**  | EfficientNetB0    | Oversampling + augmentations                 |

**Training Details**:  
- **Loss Functions**: Binary Focal Crossentropy (binary) / Focal Loss (multi-class)  
- **Optimizer**: Adam (learning rates from 1e-5 to 1e-3)  
- **Callbacks**: EarlyStopping, ReduceLROnPlateau  

---

## Results 📈
| Task                        | Accuracy   | Precision   | Recall   | F1 Score  | AUC      |
|-----------------------------|-----------:|------------:|---------:|----------:|---------:|
| **Full-slice Binary**       | 0.73 ±0.02 | 0.44 ±0.03  | 0.52 ±0.05 | 0.48 ±0.03 | 0.66 ±0.02 |
| **Full-slice Multi-class**  | 0.23 ±0.04 | 0.20 ±0.04  | 0.21 ±0.05 | 0.19 ±0.04 | 0.51 ±0.04 |
| **Zoomed-slice Binary**     | 0.72 ±0.03 | 0.45 ±0.05  | 0.63 ±0.08 | 0.52 ±0.05 | 0.75 ±0.04 |
| **Zoomed-slice Multi-class**| 0.53 ±0.02 | 0.46 ±0.04  | 0.44 ±0.03 | 0.43 ±0.03 | 0.74 ±0.02 |

### Key Observations 🔍
- **Binary classification** consistently outperforms multi-class approaches.  
- **Zoomed-slice models** often yield higher accuracy, indicating that focusing on the tumor region is beneficial.  
- **Class imbalance** severely affects multi-class performance, highlighting the need for robust balancing strategies.

---

## Known Issues

- **Oversampling Instead of Undersampling**  
  *Rationale*: When data is highly unbalanced, undersampling should be performed. However, we chose oversampling to preserve as many minority-class examples as possible. Undersampling would have reduced the diversity of the minority classes, potentially leading to worse generalization.

- **Applying SMOTE Directly to Images**  
  *Rationale*: While SMOTE is typically designed for tabular data, we adapted it for image data to increase the representation of minority classes without losing important image features. However, this approach may introduce some artifacts in the synthetic images.

---

## Conclusion 🚀
DeepBreath AI **demonstrates the feasibility** of automated lung cancer malignancy detection via deep learning. Our experiments show that **zoomed-slice binary classification** offers the most promising results. However, **multi-class classification** still faces challenges due to **imbalanced datasets** and subtle malignancy differences.

**Future Directions**:
- Expanding data with **more advanced augmentation**.  
- Investigating **transformer-based architectures**.  
- Exploring additional **balancing strategies** to tackle class imbalance.

---

## Installation & Usage ⚙️
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/DeepBreathAI.git
   cd DeepBreathAI
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebooks**  
   ```bash
   jupiter notebook.ipynb
   ```
---

## Hardware Requirements 🖥️
Our training setup was conducted on a **single NVIDIA Tesla P100 GPU** provided by Kaggle. While the code can be run on CPU, a GPU is strongly recommended to reduce training times.

**Recommended Specifications**:
- **GPU**: At least 1 × NVIDIA Tesla P100 (16GB) or equivalent/better
- **Memory**: 16GB RAM (minimum)
- **Processor**: Modern multi-core CPU
- **Storage**: 1Gb storage
---

## References 📚
For more in-depth explanations, methodology, and research citations, please refer to the associated research report.
