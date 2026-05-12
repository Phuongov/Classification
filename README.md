# 2D Alzheimer’s MRI Classification using Deep Learning

A deep learning project for multi-class Alzheimer’s disease classification from brain MRI images using transfer learning with PyTorch. This project evaluates convolutional neural network (CNN) architectures for distinguishing different stages of dementia from 2D MRI scans while addressing common challenges in medical imaging such as class imbalance and data leakage.

---

## Overview

Alzheimer’s disease is a progressive neurodegenerative disorder that affects memory and cognitive function. Early detection is important for clinical intervention and disease management.

This project develops a deep learning pipeline using transfer learning on pretrained CNN architectures to classify MRI brain images into four dementia categories:

| Class | Description |
|---|---|
| NOD | Non Demented |
| VMD | Very Mild Demented |
| MID | Mild Demented |
| MOD | Moderate Demented |

The project focuses not only on predictive performance, but also on:
- reproducible deep learning workflows,
- medical imaging evaluation metrics,
- class imbalance handling,
- and explainability for clinical interpretation.

---

## Dataset

MRI images were obtained from the Kaggle Alzheimer MRI dataset:

[Kaggle Alzheimer MRI Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)

The dataset contains labeled 2D MRI images corresponding to four dementia stages.

---

## Objectives

- Build a deep learning model for Alzheimer’s MRI classification
- Evaluate transfer learning strategies using pretrained CNN models
- Compare model performance across dementia stages
- Investigate the impact of data splitting strategies on model generalization
- Improve interpretability using visualization techniques

---

## Methods

### Deep Learning Framework
- PyTorch
- Torchvision
- Transfer Learning

### CNN Architectures Evaluated
- ResNet18
- ResNet34
- DenseNet121

### Data Processing
- Image resizing and normalization
- Data augmentation
- Stratified train/validation/test split

### Training Strategies
- Transfer learning with pretrained ImageNet weights
- Weighted cross-entropy loss for class imbalance
- Early stopping and model checkpointing

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

### Explainability
- Grad-CAM visualization
- Saliency map analysis

---

## Project Structure

```text
alzheimers-mri-classification/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
│
├── figures/
│
├── results/
│
├── requirements.txt
├── train.py
├── evaluate.py
└── README.md