# 🧠 CNV Disease Detection using OCT Images

## 📌 Project Overview

Optical Coherence Tomography (OCT) is a critical imaging technique in ophthalmology for diagnosing retinal diseases. This project applies **Deep Learning (CNN)** to automatically classify OCT images into:

- **NORMAL**
- **CNV** (Choroidal Neovascularization)

**Goal:** Reduce diagnostic workload for clinicians while maintaining high accuracy.

---

## 📑 Table of Contents

1. [Abstract](#abstract)
2. [About CNV](#about-cnv)
3. [Dataset](#dataset)
4. [Preprocessing Pipeline](#preprocessing-pipeline)
5. [Models](#models)
6. [Training Configuration](#training-configuration)
7. [Results](#results)
8. [Key Insights](#key-insights)
9. [Limitations](#limitations)
10. [Future Work](#future-work)
11. [References](#references)

---

## 📖 Abstract

This project builds and compares three Convolutional Neural Network (CNN) architectures:

- **ShallowCNN**
- **SimpleCNN**
- **ResidualCNN**

With a preprocessing pipeline including:

- Bilateral Filtering
- CLAHE (Contrast Enhancement)
- Laplacian Sharpening
- Normalization

**Results Summary:**
- ✅ ResidualCNN achieves the highest performance
- ✅ ShallowCNN is optimal for speed-constrained scenarios

---

## 🧬 About CNV

**Choroidal Neovascularization (CNV)** is a condition where abnormal blood vessels grow beneath the retina, causing:

- Fluid leakage
- Distorted retinal layers
- Dark regions in OCT images

---

## 📊 Dataset

### Data Split

| Set | Percentage | Notes |
|-----|-----------|-------|
| Train | 80% | Model training |
| Validation | 20% | Hyperparameter tuning |
| Test | Separate | Independent evaluation |

### Dataset Distribution

| Class | Train | Validation | Test |
|-------|-------|-----------|------|
| NORMAL | ~40,912 | ~10,228 | ~2,300 |
| CNV | ~29,760 | ~7,440 | ~2,300 |
---

## 🖼️ Preprocessing Pipeline

### 1️⃣ Grayscale Conversion
- OCT images are intensity-based → grayscale is sufficient

### 2️⃣ Resize
- Target size: **224×224 pixels**

### 3️⃣ Bilateral Filter
- **Parameters:** `d=9, sigmaColor=75, sigmaSpace=75`
- Reduces noise while preserving edges

### 4️⃣ CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Parameters:** `clipLimit=2.0, tileGridSize=(8,8)`
- Enhances local contrast

### 5️⃣ Laplacian Sharpening
```
output = 0.8 × image + 0.2 × |Laplacian|
```
- Highlights edges and retinal structures

### 6️⃣ Normalization
- Scale pixel values to **[0, 1]**
- Channel expansion: **(H, W) → (H, W, 1)**

---

## 🧠 Models

### 1. ShallowCNN
- **Convolution blocks:** 2
- **Speed:** Fastest ⚡
- **Feature extraction:** Lower capability
- **Use case:** Real-time systems

### 2. SimpleCNN
- **Convolution blocks:** 3
- **Performance:** Balanced
- **Use case:** Good baseline model

### 3. ResidualCNN
- **Key feature:** Residual connections
  ```
  output = x + F(x)
  ```
- **Advantages:**
  - Better gradient flow
  - Highest accuracy 🏆
  - Better convergence
- **Use case:** Maximum accuracy requirements

---

## ⚙️ Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | Adaptive learning |
| Learning Rate | 1e-3 | 0.001 |
| Batch Size | 64 | Training batch |
| Loss Function | Binary Cross Entropy | 2-class problem |
| Epochs | 10 | Training iterations |
| Hardware | GPU | If available |

---

## 📈 Results

### Loss Comparison
- **ResidualCNN** → Lowest loss ✅
- **SimpleCNN** → Stable
- **ShallowCNN** → Slight overfitting ⚠️

### Accuracy Comparison
- **ResidualCNN** → **Highest** 🥇
- **SimpleCNN** → Very close
- **ShallowCNN** → Lowest


### AUC Comparison
- **All models:** ~0.997 – 0.999
- **ResidualCNN & SimpleCNN:** ~0.999 ✅

### Model Comparison Table

| Model | Accuracy | AUC | Speed | Notes |
|-------|----------|-----|-------|-------|
| ShallowCNN | Medium | Lower | ⚡ Fastest | Slight overfitting |
| SimpleCNN | High | ~0.999 | Medium | Good baseline |
| ResidualCNN | **Highest** | ~0.999 | Slower | **Best performance** 🏆 |

---

## 🧾 Key Insights

### Main Findings
1. **Model depth** significantly affects performance
2. **Residual connections** improve convergence
3. **Preprocessing** is critical for OCT images
4. **Trade-off:** Speed vs Accuracy

### Practical Recommendations

- 🚀 **Use ShallowCNN** for real-time systems (embedded devices, edge computing)
- 🎯 **Use ResidualCNN** for maximum accuracy (clinical diagnosis)
- ⚖️ **Use SimpleCNN** for balanced scenarios (general applications)

---

## ⚠️ Limitations

- ❌ Only binary classification (NORMAL vs CNV)
- ❌ No data augmentation (hardware limitation)
- ❌ Dataset imbalance and split limitations
- ❌ Models are relatively simple compared to State-of-the-Art (SOTA)

---

## 🚀 Future Work

- [ ] Multi-class classification (DME, DRUSEN, etc.)
- [ ] Advanced data augmentation for medical images
- [ ] Use real-world clinical datasets
- [ ] Deploy real-time diagnostic system
- [ ] Model ensemble techniques
- [ ] Explainability & interpretability (Grad-CAM, etc.)

---

## 📚 References {#references}

### Research Papers
- **Kermany et al. (2018)** - Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-ray Images
  - DOI: https://doi.org/10.17632/rscbjbr9sj.3

### Datasets
- **Kaggle OCT Dataset**
  - https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8

### Libraries & Tools
- 🔗 [PyTorch](https://pytorch.org/)
- 🔗 [OpenCV](https://opencv.org/)
- 🔗 [NumPy](https://numpy.org/)
- 🔗 [Scikit-learn](https://scikit-learn.org/)

---

## 🙏 Acknowledgements

**Special thanks to:**
- 👨‍🏫 **Thầy Ngô Quốc Việt** for guidance and supervision
- 💻 **Open-source communities:**
  - PyTorch Foundation
  - OpenCV Community
  - NumPy & SciPy Contributors
  - Scikit-learn Team

---

## 📌 Conclusion

This project demonstrates that:

✅ **ResidualCNN** achieves the best overall performance  
✅ **ShallowCNN** provides strong speed-accuracy trade-off  
✅ **Deep learning** is highly effective for OCT image classification  
✅ **Strong potential** for real-world clinical applications  

---

*Last Updated: 2026*  
*Project Status: Complete* ✨