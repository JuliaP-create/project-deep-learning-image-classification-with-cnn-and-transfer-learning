![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project | Deep Learning: Image Classification using CNN and Transfer Learning

## Task Description

In this project, students will first build a **Convolutional Neural Network (CNN)** model from scratch to classify images from a given dataset into predefined categories. Then, they will implement a **transfer learning approach** using a pre-trained model. Finally, students will **compare the performance** of the custom CNN and the transfer learning model based on evaluation metrics and analysis.

## Project Overview

This project systematically explores CNN architectures for CIFAR-10 image classification, progressing from simple 5-layer networks to advanced transfer learning with EfficientNetV2B0. The final optimized model achieves **93.97% test accuracy**, demonstrating a +6.27% improvement over custom ResNet-20 architecture.

**Author:** Julia Parnis
**Date:** December 2025  
**Framework:** TensorFlow/Keras
> **Note:** This represents my individual work from a collaborative team project.  
> All models, training, and analysis in this notebook are my original implementation.

---

## Dataset

**CIFAR-10** consists of 60,000 32×32 color images across 10 classes:
- **Training set:** 50,000 images (40,000 training + 10,000 validation after split)
- **Test set:** 10,000 images
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Dataset loaded directly from `tf.keras.datasets.cifar10`.

---

## Project Structure
project/
├── image_classification_project_JP.ipynb   # Main notebook with all experiments
├── requirements.txt                        # Python dependencies
├── Image_classification_CNN.pdf            # PDF of Power Point presentation of the project
├── models/                                 # Saved model files
│ ├── resnet20_model.h5                     # ResNet-20 (1 MB)
│ ├── vgg10_optimized.h5                    # VGG-10 optimized (4 MB)
│ └── efficientnet_224x224_dropout02.h5     # Final transfer model (43 MB)
└── README.md                               # This file


---

## Methodology

### Step 1: Data Exploration
- Visualized sample images from each class
- Analyzed class distribution (balanced: 6,000 images per class)
- Examined image properties 

### Step 2: Data Preprocessing
- Images were normalized for all the models by dividing by 255, but kept in original [0, 255] range for EfficientNet compatibility
- Data augmentation applied: rotation (15°), horizontal flip, zoom (10%), width/height shifts (10%)
- 80/20 train-validation split using `ImageDataGenerator`

### Step 3: Custom CNN Models

**3.1 Simple 5-Layer CNN**
- Architecture: Conv→MaxPool→Conv→MaxPool→Dense
- Parameters: 316K
- Result: 67.5% accuracy (severe overfitting)

**3.2 VGG-10 Model**
- Architecture: 3×(Conv→Conv→MaxPool)→Dense
- Parameters: 403K
- Iterations tested: baseline, + callbacks, + data augmentation
- Best result: 82.1% accuracy with data augmentation, no overfitting

**3.3 ResNet-20 Model**
- Architecture: Conv→9×(ResidualBlock)→GlobalAvgPool→Dense
- Parameters: 275K (most efficient)
- Features: Residual connections, batch normalization
- Result: **87.7% accuracy** with +3.1% overfitting gap

### Step 4: Transfer Learning

**4.1 Initial Transfer Model (EfficientNetV2B0)**
- Input resolution: 96×96
- Training: Two-phase (30+30 epochs)
- Dropout: 0.3
- Fine-tuned layers: 30 (last 15%)
- Result: 90.4% accuracy

**4.2 Optimized Transfer Model (EfficientNetV2B0)**
- Input resolution: 224×224 (native)
- Training: Two-phase (40+40 epochs)
- Dropout: 0.2
- Fine-tuned layers: 50 (last 25%)
- Result: **93.97% accuracy** with near-zero overfitting (-0.01%)

---

## Results Summary

| Model              | Test Accuracy| Test Loss| Overfitting Gap|Parameters                 |
|--------------------|--------------|----------|----------------|---------------------------|
| 5-Layer CNN        | 67.47%       | 1.104    | +17.8%         |316K                       |
| VGG-10 (optimized) | 82.08%       | 0.543    | -1.8%          |403K                       |
| ResNet-20          | 87.70%       | 0.389    | +3.1%          |275K                       |
| EfficientNetV2     | 90.42%       | 0.275    | -4.5%          |6.26M (334K trainable)     |
| (96x96)            |
| **EfficientNetV2** |
|**(224×224)**       | **93.97%**   | **0.174**| **-0.01%**     |**6.26M (334K trainable)** |

### Key Findings
**Impact Analysis:**
1. **Architecture Design (5-Layer → VGG-10):** +14.61% - Largest single improvement
2. **Residual Connections (VGG-10 → ResNet-20):** +5.62% - Enabled deeper, more efficient training
3. **Input Resolution (96×96 → 224×224):** +3.55% - Critical for transfer learning
4. **Transfer Learning (ResNet-20 → EfficientNet):** +2.72% - Pre-trained features advantage

**Overall Progression:**
- Simple CNN (67.5%) → VGG-10 (82.1%) → ResNet-20 (87.7%) → Transfer Learning (93.97%)
- Total improvement: **+26.5 percentage points**

---

## Requirements

See `requirements.txt` for details. Key packages:
- TensorFlow 2.15+
- NumPy
- Matplotlib
- scikit-learn

---

## How to Run

### 1. Install Dependencies
pip install -r requirements.txt


### 2. Run Notebook
jupyter notebook image_classification_project_JP.ipynb


### 3. Hardware Requirements
- **Recommended:** GPU (NVIDIA A100 or similar)
- **RAM:** 12GB+ (High RAM recommended for 224×224 training)
- **Training time:** 
  - Simple models: 5-30 minutes
  - ResNet-20: ~40-50 minutes
  - Transfer learning: ~30-50 minutes (depends on resolution)

---

## Model Files

Due to file size limitations:
-  **Included:** - ResNet-20 (4 MB), VGG-10 optimized (5 MB), EfficientNetV2 model (43 MB)
-  **Not included:** - Weights-only files (`*.weights.h5`)
                     - Training history/metrics artifacts (`history_*.pkl`, `metrics_*.pkl`) 
-  **To reproduce:** Run notebook cells - models will be trained and saved locally

---

## Evaluation Metrics

All models evaluated using:
- Accuracy, Loss, overfitting gap
- Training curves (accuracy and loss over epochs)

In addition, optimized ResNet and optimized EfficientNet models are also evaluated using:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-class accuracy analysis
- Calibration plots (confidence vs actual accuracy)
- Mismatched images

---

## Conclusions

Transfer learning with EfficientNetV2B0 achieved 93.97% accuracy, significantly outperforming custom architectures. Key success factors:
1. Using pre-trained model at native resolution (224×224)
2. Two-phase training (feature extraction → fine-tuning)
3. Systematic hyperparameter optimization (dropout, epochs, fine-tuned layers)

The results validate that properly optimized transfer learning surpasses even well-designed custom architectures (ResNet-20) trained from scratch.

---

## References

- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- EfficientNetV2: https://arxiv.org/abs/2104.00298
- TensorFlow Transfer Learning Guide: https://www.tensorflow.org/tutorials/images/transfer_learning

---

## License

This project is for educational purposes as part of Ironhack's Data Analytics bootcamp.

