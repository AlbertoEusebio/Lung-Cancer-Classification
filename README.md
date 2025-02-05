# DeepBreath AI - Lung Cancer Malignancy Score

## Authors
Sofia Brunori, Alberto Eusebio, Alessandro Tognotti, Andrea Tomasella

## Introduction
Lung cancer (LC) remains one of the leading causes of death worldwide, with early diagnosis being critical for improving survival rates. Computed Tomography (CT) scans serve as a primary imaging modality for LC diagnosis, but manual analysis is time-consuming and prone to errors.

DeepBreath AI aims to develop machine learning classifiers for lung cancer malignancy detection, assessing malignancy scores and confidence estimation across different input types. The classification task includes:
- Binary classification (Benign: Classes 1, 2, 3 vs. Malignant: Classes 4, 5)
- Multi-class malignancy classification (Classes 1-5)
- Analysis of model performance using full-slice and zoomed-slice CT scans

## Dataset
The dataset comprises CT scan slices from **2,363 patients**, each represented by:
- A **full CT slice** (512x512 int16 matrix)
- A **zoomed-in nodule slice** with varying dimensions

All slices are stored in **NRRD format** with Hounsfield Unit (HU) values.

### Class Distribution
- **Binary classification**: 1,793 benign vs. 570 malignant samples
- **Multi-class classification**: Imbalanced distribution across 5 malignancy scores

## Methods
### Data Preprocessing
1. **Normalization**: Min-max normalization applied to scale values within [0,1].
2. **HU Windowing Transformation**: Applied to enhance lung tumor visibility.
   - Window width: **1300**, Window level: **-220**
3. **Augmentation**: Applied based on classification task.
4. **Balancing Techniques**:
   - **Binary classification**: Random oversampling
   - **Multi-class classification**: SMOTE + class weight balancing

### Model Architectures
EfficientNet-based architectures were employed for different classification tasks:
- **Full-slice Binary Classification**: EfficientNetV2S with fine-tuning of last 10 layers
- **Full-slice Multi-class Classification**: EfficientNetB1 with class balancing
- **Zoomed-slice Binary Classification**: EfficientNetV2S with SMOTE augmentation
- **Zoomed-slice Multi-class Classification**: EfficientNetB0 with oversampling and augmentation

**Training Details**:
- **Loss Function**: Focal Loss (multi-class), Binary Focal Crossentropy (binary)
- **Optimization**: Adam Optimizer with learning rates from 1e-5 to 1e-3
- **Callbacks**: EarlyStopping and ReduceLrOnPlateau applied for stability

## Results
| Task                     | Accuracy  | Precision | Recall  | F1 Score | AUC  |
|--------------------------|-----------|-----------|---------|----------|------|
| Full-slice Binary       | 0.73 ±0.02 | 0.44 ±0.03 | 0.52 ±0.05 | 0.48 ±0.03 | 0.66 ±0.02 |
| Full-slice Multi-class  | 0.23 ±0.04 | 0.20 ±0.04 | 0.21 ±0.05 | 0.19 ±0.04 | 0.51 ±0.04 |
| Zoomed-slice Binary     | 0.72 ±0.03 | 0.45 ±0.05 | 0.63 ±0.08 | 0.52 ±0.05 | 0.75 ±0.04 |
| Zoomed-slice Multi-class| 0.53 ±0.02 | 0.46 ±0.04 | 0.44 ±0.03 | 0.43 ±0.03 | 0.74 ±0.02 |

### Key Findings
- **Binary classification tasks performed better** than multi-class classification.
- **Zoomed-slice models outperformed full-slice models** due to improved tumor focus.
- **Class imbalance significantly impacted performance**, requiring oversampling and weighted loss functions.

## Conclusion
DeepBreath AI demonstrates the feasibility of lung cancer malignancy detection using deep learning on CT scans. The results suggest that **zoomed-slice binary classification yields the most reliable performance**, while multi-class classification remains a challenge due to dataset imbalance and tumor visibility constraints.

Further improvements can be made through:
- More advanced augmentation techniques
- Exploration of transformer-based architectures
- Additional balancing strategies for multi-class classification

## Installation & Usage
### Prerequisites
- Python 3.8+
- TensorFlow, Keras
- OpenCV, NumPy, Scikit-learn, Pandas, Matplotlib
- SimpleITK (for NRRD format handling)

## References
Refer to the full project documentation for additional details and cited research papers.

