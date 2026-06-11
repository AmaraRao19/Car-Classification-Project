# 🚗 Car Classification Project (Deep Learning CNN)

---

## 📌 Project Overview

This project is an **advanced deep learning-based Car Classification system** that identifies different car brands and models from images.

It uses a **hybrid CNN architecture** combining:

- Multiscale feature extraction (1×1, 3×3, 5×5 convolutions)
- Residual connections (ResNet-style)
- Attention mechanism (SE-Net style)

This allows the model to learn **fine-grained and high-level features** for accurate classification.

---

## 🎯 Objective

To build a highly accurate and robust image classification system that can recognize car types using advanced CNN architectures with feature enhancement techniques.

---

## 📂 Dataset Structure

train/
├── Audi
├── Swift
├── Toyota Innova

test/
├── Audi
├── Swift
├── Toyota Innova

---

## ⚙️ Project Flow (Pipeline)

### 1. Import Libraries
- TensorFlow (Deep Learning)
- NumPy (Numerical operations)
- Matplotlib / Seaborn (Visualization)
- Scikit-learn (Evaluation metrics)

---

### 2. Dataset Loading
Used:
tf.keras.preprocessing.image_dataset_from_directory

Automatically:
- Assigns labels
- Reads images
- Resizes images
- Creates batches

---

### 3. Train / Validation Split
- Train: 80%
- Validation: 20%

Used:
validation_split=0.2

---

### 4. Data Preprocessing
Rescaling(1./255)

✔ Converts pixel values to range 0–1

---

### 5. Data Augmentation
- RandomFlip
- RandomRotation
- RandomZoom
- RandomContrast

✔ Prevents overfitting  
✔ Improves generalization  

---

## 🧠 Model Architecture

### Convolutional Feature Extraction
- Conv2D
- MaxPooling2D

✔ Extract edges, textures, shapes

---

### Multiscale Feature Block
- 1×1 Conv
- 3×3 Conv
- 5×5 Conv
- Concatenation

✔ Captures small, medium, and large features

---

### Residual Block
- Skip connections using Add()

✔ Solves vanishing gradient problem  
✔ Improves deep learning stability  

---

### Attention Block
- GlobalAveragePooling
- Dense layers
- Sigmoid activation
- Feature weighting

✔ Focuses on important features  

---

## 🏗️ Architecture Diagram

(Insert your CNN architecture image here)

---

## 🧾 Classification Head

- GlobalAveragePooling2D
- Dropout
- Dense(256)
- Dense(NUM_CLASSES, Softmax)

✔ Outputs probability for each class

---

## ⚙️ Compilation Settings

optimizer = Adam
loss = sparse_categorical_crossentropy
metrics = accuracy

✔ Adam → fast optimizer  
✔ Loss → multi-class classification  
✔ Accuracy → performance metric  

---

## 🏋️ Training Process

- model.fit()
- EarlyStopping
- ReduceLROnPlateau

✔ Prevents overfitting  
✔ Auto learning rate adjustment  

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

✔ Measures model performance properly  

---

## 📈 Visualization

- Accuracy graph
- Loss graph
- Confusion matrix heatmap

✔ Shows training behavior  

---

## 💾 Model Saving

model.save("car_classification_model.h5")

✔ Saves trained model for reuse  

---

## 🚀 Technologies Used

- Python 🐍
- TensorFlow / Keras 🤖
- NumPy 📊
- Matplotlib 📈
- Seaborn 📉
- Scikit-learn ⚙️

---
