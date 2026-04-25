# 🚀 RSWaD-Edge-AI  
### Real-Time Silicon Wafer Defect Detection using Hybrid Edge Intelligence  

---

## 🌌 Abstract

RSWaD-Edge-AI presents a real-time silicon wafer defect detection framework designed for semiconductor manufacturing environments. The system integrates deep learning-based classification with edge inference to achieve an optimal trade-off between computational efficiency and prediction accuracy.

A hybrid inference strategy is employed, where MobileNetV3 performs fast initial predictions, and MobileNetV2 is selectively invoked when prediction confidence falls below a defined threshold. This adaptive approach ensures both low latency and high reliability. The system further evaluates inference latency and system-level performance metrics, making it suitable for deployment in edge-based industrial settings.

---

## 🧠 Core Idea

The core concept of this system is **adaptive hybrid inference** using two convolutional neural network models:

- **MobileNetV3** performs initial inference due to its lightweight and fast nature  
- A **confidence score** is computed using entropy-based evaluation  
- If the confidence is **above threshold**, the prediction is accepted  
- If the confidence is **below threshold**, inference is repeated using **MobileNetV2**  
- The final output is selected based on this decision logic  

This strategy ensures:
- Faster predictions in most cases  
- Improved accuracy when uncertainty is detected  
- Efficient utilization of computational resources  

---

## ⚙️ Methodology

### 🔹 Data Processing

- Input wafer images are resized to **224 × 224 resolution**  
- Images are converted to **RGB format**  
- Pixel values are normalized to the range **[-1, 1]**  
- Preprocessing ensures compatibility with trained CNN models  

---

### 🔹 Model Design

#### **MobileNetV3**
- Used for **initial inference**
- Optimized for **low latency and edge deployment**
- Provides fast predictions with reasonable accuracy  

#### **MobileNetV2**
- Used as a **fallback model**
- Activated only when confidence is low  
- Provides **higher reliability** in uncertain cases  

---

### 🔹 Confidence Computation

Confidence is calculated using **normalized entropy**, which measures prediction uncertainty:

H(p) = -∑ p log(p) and 
Confidence = 1 - (H(p) / log(N))


Where:
- `p` = predicted probability distribution  
- `N` = number of classes  

Lower entropy → higher confidence  
Higher entropy → uncertain prediction  

---

### 🔹 Adaptive Switching Logic

- If **confidence ≥ threshold** → accept MobileNetV3 output  
- If **confidence < threshold** → invoke MobileNetV2  
- Final prediction is selected accordingly  

This ensures:
- Fast inference for confident cases  
- Accurate inference for uncertain cases  

---

### 🔹 Edge Inference

- Implemented using **ONNX Runtime**  
- Supports **CPU and GPU execution**  
- Designed for **real-time edge deployment**  
- Measures inference latency dynamically  

---

## 📊 Performance Metrics

The system evaluates and reports the following:

- **Inference Latency** (ms)  
- **Selected Model** (V2 or V3)  
- **Confidence Score (%)**  
- **CPU Power Consumption (W)** *(if available)*  
- **GPU Power Consumption (W)** *(if available)*  

These metrics provide insight into both model performance and system efficiency.

---

## 🎨 Visualization Interface

The system includes a web-based interface for real-time interaction and visualization:

- Displays uploaded wafer image  
- Shows predicted defect category  
- Provides confidence score and selected model  
- Displays latency and system performance metrics  

The interface follows a structured layout:
- Central region for wafer visualization and prediction  
- Surrounding panels for performance metrics  

---

## 📦 Dataset

The model is trained and evaluated on wafer map data containing multiple defect categories:

- Scratch  
- Donut  
- Edge-Loc  
- Edge-Ring  
- Center  
- Near-Full  
- Random  

These categories represent common defect patterns in semiconductor wafers.

---

## 🚀 Key Contributions

- Hybrid inference using **MobileNetV2 and MobileNetV3**  
- **Entropy-based confidence evaluation** mechanism  
- **Adaptive model switching** for improved efficiency  
- Real-time inference suitable for **edge deployment**  
- Integrated monitoring of **latency and system performance**  

---

## 🔬 Applications

- Semiconductor wafer inspection  
- Defect classification in fabrication pipelines  
- Edge AI systems for industrial automation  
- Real-time quality monitoring  

---

## 🧠 Conclusion

RSWaD-Edge-AI demonstrates an efficient hybrid edge AI framework for wafer defect detection. By combining fast and accurate models with a confidence-based switching mechanism, the system achieves a balance between performance and reliability.

This approach enables real-time inference suitable for edge environments while maintaining prediction accuracy, making it a practical solution for semiconductor manufacturing applications.

---
