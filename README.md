#🚗 Car Classification Project (Deep Learning CNN)
📌 Project Overview

This project is an advanced deep learning-based Car Classification system that identifies different car brands and models from images.

It uses a hybrid CNN architecture combining:

Multiscale feature extraction (1×1, 3×3, 5×5 convolutions)
Residual connections (ResNet-style)
Attention mechanism (SE-Net style)

This allows the model to learn fine-grained and high-level features for accurate classification.

🎯 Objective

To build a highly accurate and robust image classification system that can recognize car types using advanced CNN architectures with feature enhancement techniques.

📂 Dataset Structure
train/
   ├── Audi
   ├── Swift
   ├── Toyota Innova

test/
   ├── Audi
   ├── Swift
   ├── Toyota Innova

✔ Automatically labeled using folder names
✔ Images resized and batched using TensorFlow utilities

⚙️ Project Flow (Pipeline)
🟢 1. Import Libraries

Setup environment using:

TensorFlow (Deep Learning)
NumPy (Numerical operations)
Matplotlib / Seaborn (Visualization)
Sklearn (Evaluation metrics)
🟢 2. Dataset Loading

Used:

tf.keras.preprocessing.image_dataset_from_directory

✔ Automatically:

Assigns labels
Reads images
Resizes images
Creates batches
🟢 3. Train / Validation Split
Train: 80%
Validation: 20%

Used:

validation_split=0.2
🟢 4. Data Preprocessing

Images normalized using:

Rescaling(1./255)

✔ Converts pixel range to 0–1 for better training stability

🟢 5. Data Augmentation

Applied transformations:

RandomFlip
RandomRotation
RandomZoom
RandomContrast

✔ Prevents overfitting
✔ Improves generalization

🧠 Model Architecture
🔷 Convolutional Feature Extraction
Conv2D layers
MaxPooling layers
✔ Extract edges, textures, and shapes
🔷 Multiscale Feature Block
1×1 Convolution
3×3 Convolution
5×5 Convolution
Concatenation

✔ Captures:

Small features (lights, edges)
Medium features (windows, wheels)
Large features (entire car shape)
🔷 Residual Block (ResNet Style)
Skip connections using Add()

✔ Solves vanishing gradient problem
✔ Improves deep network performance

🔷 Attention Block (SE-Net Style)
GlobalAveragePooling
Dense layers
Sigmoid activation
Feature re-weighting

✔ Focuses on important regions
✔ Reduces background noise

🏗️ Architecture Diagram

Note: Your actual model is more advanced (CNN + Multiscale + Residual + Attention)

🧾 Classification Head
GlobalAveragePooling2D
Dropout (regularization)
Dense(256)
Dense(NUM_CLASSES, Softmax)

✔ Outputs probability for each car class

⚙️ Compilation Settings
optimizer = Adam
loss = sparse_categorical_crossentropy
metrics = accuracy

✔ Adam → fast convergence
✔ Loss → multi-class classification
✔ Accuracy → performance measure

🏋️ Training Process

Used:

model.fit()
EarlyStopping
ReduceLROnPlateau

✔ Prevents overfitting
✔ Automatically adjusts learning rate

📊 Evaluation Metrics
Accuracy
Precision
Recall
F1 Score
Confusion Matrix

✔ Measures real-world performance

📈 Visualization
Accuracy vs Epoch graph
Loss vs Epoch graph
Confusion Matrix heatmap

✔ Shows model learning behavior

💾 Model Saving
model.save("car_classification_model.h5")

✔ Saves trained model for reuse
✔ No need to retrain

🚀 Technologies Used
Python 🐍
TensorFlow / Keras 🤖
NumPy 📊
Matplotlib 📈
Seaborn 📉
Scikit-learn ⚙️
