# Comparative Analysis of Linear and Non-Linear Dimensionality Reduction for Face Recognition and Verification

[![Python](https://img.shields.io/badge/python-3.14+-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-orange)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-blueviolet)](https://pandas.pydata.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)](https://pytorch.org/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7+-green)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-yellowgreen)](https://scikit-learn.org/)

- **Students:** Roberto Carriero, Massimiliano Leone
- **Course:** Statistical and Mathematical Methods for Artificial Intelligence - 2025/2026
- **Program:** M.Sc. Computer Science Engineering (Artificial Intelligence and Data Science) - Politecnico di Bari

---

## Overview

This project provides an empirical comparison between **PCA** (Principal Component Analysis) and **Autoencoders** for dimensionality reduction in face recognition and verification tasks. The study evaluates both linear and non-linear approaches on the Labeled Faces in the Wild (LFW) dataset, analyzing their effectiveness for classification and identity verification.

## Objectives

- Compare linear (PCA) vs. non-linear (Autoencoder) dimensionality reduction
- Evaluate classification performance using SVM and Neural Network classifiers
- Assess face verification through cosine similarity and ROC analysis
- Analyze reconstruction quality and feature discriminability

## Methodology

1. **Data Preprocessing**: Load LFW dataset, normalize pixel values, apply z-score standardization
2. **Dimensionality Reduction**: Train PCA (via SVD) and Autoencoder models across multiple latent dimensions
3. **Classification**: Grid search hyperparameter tuning with 5-fold stratified cross-validation for SVM and NN classifiers
4. **Verification**: Generate genuine/impostor pairs, compute cosine similarity, evaluate AUC and EER
5. **Visualization**: t-SNE projections, confusion matrices, ROC curves, reconstruction comparisons

## Project Structure

```
face-recognition-dimensionality-reduction/
├── config.py              # Configuration and hyperparameters
├── main.py                # Main pipeline execution
├── requirements.txt       # Dependencies
├── src/
│   ├── preprocessing.py   # Data loading and preprocessing
│   ├── pca.py             # PCA implementation using SVD
│   ├── autoencoder.py     # Neural network autoencoder
│   ├── classification.py  # SVM and NN classifiers
│   ├── verification.py    # Face verification evaluation
│   ├── metrics.py         # Performance metrics computation
│   └── utils.py           # Visualization utilities
└── results/
    ├── figures/           # Generated plots
    └── models/            # Saved trained models
```

## Requirements

- Python 3.14+
- numpy >= 1.24.0, < 2.0.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- torch >= 2.0.0, < 2.11.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- joblib >= 1.3.0
- pillow >= 10.0.0
- tqdm >= 4.65.0

## Installation

```bash
git clone https://github.com/carrieroroberto/face-recognition-dimensionality-reduction.git
cd face-recognition-dimensionality-reduction
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:

```bash
python main.py
```

The script executes:
1. Dataset loading and preprocessing
2. Ablation study for PCA and Autoencoder (dimensions: 25, 50, 75, 100, 150)
3. Classifier training with hyperparameter optimization
4. Face verification evaluation
5. Results export and model saving

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_FACES_PER_PERSON` | 70 | Minimum images per identity |
| `RESIZE_RATIO` | 0.5 | Image downscaling factor |
| `TEST_SIZE` | 0.2 | Train-test split ratio |
| `COMPONENTS_RANGE` | [25, 50, 75, 100, 150] | PCA directions/AE latent dimensions to evaluate |
| `AE_EPOCHS` | 200 | Autoencoder training epochs |
| `CV_FOLDS` | 5 | Cross-validation folds |

## Output

- **Models**: Saved PCA (.joblib) and Autoencoder (.pt) models
- **Metrics**: Classification results exported to CSV
- **Figures**: Eigenfaces, ROC curves, confusion matrices, t-SNE plots, verification analysis

### Classification Metrics

| Model | Features | Accuracy | F1 (Macro) | F1 (Weighted) | ROC AUC | CV Score |
|-------|----------|----------|------------|---------------|---------|----------|
| **NN** | **PCA** | **0.8488** | **0.8072** | **0.8518** | **0.9809** | **0.8592** |
| NN | Autoencoder | 0.8178 | 0.7849 | 0.8200 | 0.9720 | 0.7981 |
| SVM | PCA | 0.8062 | 0.7423 | 0.8015 | 0.9800 | 0.8146 |
| SVM | Autoencoder | 0.7984 | 0.7474 | 0.7939 | 0.9710 | 0.7786 |
