# Deep Learning Project: CIFAR-10 Classification with ResNet50V2

![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-green)

**Advanced deep learning classification project** that applies transfer learning using **ResNet50V2** on the **CIFAR-10** dataset. The project implements a custom training loop with `tf.GradientTape`, a structured two-phase training strategy (feature extraction → fine-tuning), comprehensive evaluation, and comparison of different training regimes.

---

## ✨ Key Features

- Transfer learning with pre-trained ResNet50V2 (ImageNet weights)  
- Custom training loop using `tf.GradientTape` for full control  
- Two-phase training strategy:  
  - Phase 1: Freeze base model → train only top classifier  
  - Phase 2: Unfreeze selected layers → fine-tune with low learning rate  
- Comprehensive evaluation: accuracy, precision, recall, F1-score, confusion matrix  
- Learning curves, loss/accuracy plots per phase  
- Model checkpointing and best model saving  
- Dark cyber-themed presentation slides included  
- Reproducible Jupyter notebook workflow  

---

## 🚀 Quick Start

1. Clone the repository  
   ```bash
   git clone https://github.com/ElmoGaber/DeepLearning_Project.git
   cd DeepLearning_Project
   cd DeepLearning_Assig_1
   ```
