# 🚗 Car Classification using MobileNetV2 with Multiscale, Residual, and Attention Networks

## 📌 Project Overview

This project presents an advanced deep learning-based car image classification system capable of recognizing different car brands and models from images.

The proposed architecture combines the power of **Transfer Learning** with custom-designed deep learning blocks to improve feature extraction and classification performance.

Instead of relying solely on a pretrained network, the model integrates:

* ✅ MobileNetV2 (ImageNet Pretrained Backbone)
* ✅ Multiscale Feature Extraction Block
* ✅ Residual Learning Block
* ✅ Squeeze-and-Excitation (SE) Attention Mechanism
* ✅ Transfer Learning and Fine-Tuning

These components work together to capture both low-level and high-level visual features, resulting in a more robust and accurate classification system.

---

# 🎯 Project Objectives

The primary objectives of this project are:

* Build a high-performance car classification model.
* Leverage transfer learning using MobileNetV2.
* Design custom CNN blocks for enhanced feature learning.
* Improve classification accuracy through attention mechanisms.
* Reduce overfitting using data augmentation and regularization techniques.
* Visualize and evaluate model performance using multiple metrics.

---

# 📂 Dataset Structure

The dataset is organized into training and testing directories, where each folder represents a specific car category.

```text
Cars Dataset/
│
├── train/
│   ├── Audi
│   ├── Swift
│   ├── Toyota Innova
│   └── ...
│
└── test/
    ├── Audi
    ├── Swift
    ├── Toyota Innova
    └── ...
```

TensorFlow automatically assigns labels based on folder names.

---

# ⚙️ Data Preprocessing Pipeline

## 1. Dataset Loading

The dataset is loaded using:

```python
tf.keras.preprocessing.image_dataset_from_directory()
```

This automatically:

* Reads images from folders
* Assigns labels
* Resizes images
* Creates batches
* Generates TensorFlow datasets

---

## 2. Train-Validation Split

Training data is split into:

* Training Set → 80%
* Validation Set → 20%

```python
validation_split=0.2
```

This allows monitoring model performance during training.

---

## 3. Image Resizing

All images are resized to:

```text
224 × 224 × 3
```

to match MobileNetV2 input requirements.

---

## 4. Data Augmentation

To improve generalization and reduce overfitting, the following augmentation techniques are applied:

* Random Horizontal Flip
* Random Rotation
* Random Zoom
* Random Contrast Adjustment

Benefits:

* Increases dataset diversity
* Improves robustness
* Prevents memorization of training samples

---

# 🧠 Proposed Deep Learning Architecture

The model combines MobileNetV2 with custom-designed feature enhancement modules.

## Architecture Flow

```text
Input (224×224×3)
        │
        ▼
Data Augmentation
        │
        ▼
MobileNetV2 Backbone
(ImageNet Pretrained)
        │
        ▼
Multiscale Feature Extraction Block
 ├── 1×1 Convolution
 ├── 3×3 Convolution
 └── 5×5 Convolution
        │
        ▼
Concatenation Layer
        │
        ▼
Residual Block
        │
        ▼
SE Attention Block
        │
        ▼
Global Average Pooling
        │
        ▼
Dropout (0.5)
        │
        ▼
Dense Layer (512)
        │
        ▼
Batch Normalization
        │
        ▼
Dropout (0.3)
        │
        ▼
Dense Layer (NUM_CLASSES)
        │
        ▼
Softmax Classification
```

---

# 🔍 MobileNetV2 Backbone

MobileNetV2 is used as the primary feature extractor.

### Advantages

* Lightweight architecture
* Fast training and inference
* Pretrained on ImageNet
* Excellent feature extraction capability
* Suitable for transfer learning applications

The ImageNet-trained weights provide rich visual representations that significantly improve classification performance.

---

# 🔍 Multiscale Feature Extraction Block

This block processes features using multiple convolution kernels simultaneously.

### Components

* 1×1 Convolution
* 3×3 Convolution
* 5×5 Convolution

### Benefits

* Captures local details
* Learns medium-scale patterns
* Extracts large contextual features
* Improves feature diversity

Outputs from all branches are concatenated into a unified feature representation.

---

# 🔍 Residual Block

The Residual Block introduces skip connections inspired by ResNet.

### Benefits

* Reduces vanishing gradient issues
* Improves information flow
* Enables deeper feature learning
* Stabilizes training

Residual learning helps preserve important information from previous layers.

---

# 🔍 SE Attention Block

The Squeeze-and-Excitation (SE) Attention Module enables the network to focus on the most informative channels.

### Process

1. Global Average Pooling
2. Channel Compression
3. Channel Excitation
4. Feature Recalibration

### Benefits

* Highlights important features
* Suppresses irrelevant information
* Improves classification accuracy
* Enhances feature representation

---

# 🧾 Classification Head

The final classification module consists of:

```text
Global Average Pooling
↓
Dropout (0.5)
↓
Dense (512)
↓
Batch Normalization
↓
Dropout (0.3)
↓
Softmax Layer
```

### Purpose

* Reduces feature dimensions
* Prevents overfitting
* Improves generalization
* Produces class probabilities

---

# ⚙️ Model Compilation

The model is compiled using:

```python
optimizer = Adam(learning_rate=1e-4)
loss = sparse_categorical_crossentropy
metrics = ["accuracy"]
```

### Why Adam?

* Adaptive learning rate
* Faster convergence
* Widely used in deep learning applications

---

# 🏋️ Training Strategy

The model is trained using:

```python
model.fit()
```

with the following callbacks:

### EarlyStopping

Stops training when validation performance stops improving.

Benefits:

* Prevents overfitting
* Saves training time

### ReduceLROnPlateau

Automatically reduces learning rate when validation accuracy stagnates.

Benefits:

* Better convergence
* Improved optimization

---

# 📊 Evaluation Metrics

The model is evaluated using:

### Accuracy

Measures overall prediction correctness.

### Precision

Measures prediction quality.

### Recall

Measures ability to identify all relevant classes.

### F1 Score

Balances precision and recall.

### Confusion Matrix

Visualizes class-wise prediction performance.

---

# 📈 Visualization

The project generates:

### Training Accuracy Curve

Tracks accuracy improvement during training.

### Validation Accuracy Curve

Monitors generalization capability.

### Training Loss Curve

Shows optimization progress.

### Validation Loss Curve

Helps detect overfitting.

### Confusion Matrix Heatmap

Displays class-wise classification results.

---

# 💾 Model Saving

The trained model is saved as:

```python
model.save("car_classification_mobilenetv2.h5")
```

This allows future deployment and inference without retraining.

---

# 🚀 Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook / Kaggle

---

# 🏆 Key Features

* Transfer Learning with MobileNetV2
* Custom Multiscale Feature Extraction
* Residual Learning
* SE Attention Mechanism
* Data Augmentation
* Early Stopping
* Learning Rate Scheduling
* Comprehensive Evaluation Metrics
* Confusion Matrix Visualization
* Model Saving and Deployment Ready

---

# 📌 Conclusion

This project demonstrates a hybrid deep learning architecture that combines the efficiency of MobileNetV2 with custom feature enhancement modules such as Multiscale Convolution, Residual Learning, and SE Attention. The resulting model achieves robust car classification performance while maintaining computational efficiency, making it suitable for real-world intelligent transportation and vehicle recognition applications.
