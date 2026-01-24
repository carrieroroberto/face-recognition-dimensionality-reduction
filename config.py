"""
This module centralizes all hyperparameters, paths, and settings used throughout
the project. It ensures reproducibility by setting random seeds and provides
a single source of truth for experimental parameters.
"""

import os
import numpy as np
import torch
import random

# =============================================================================
# REPRODUCIBILITY SETTINGS
# Set random seeds across all libraries to ensure reproducible results
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# PATH CONFIGURATION
# Define output directories for models and visualizations
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "results", "models")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "figures")

# =============================================================================
# DATASET PARAMETERS
# Configuration for loading and splitting the LFW dataset
# =============================================================================
MIN_FACES_PER_PERSON = 70  # Minimum number of images required per person
RESIZE_RATIO = 0.5  # Image downscaling factor for faster processing
TEST_SIZE = 0.2  # Proportion of data reserved for testing

# =============================================================================
# DIMENSIONALITY REDUCTION PARAMETERS
# Range of component/dimension values for ablation study
# =============================================================================
COMPONENTS_RANGE = [25, 50, 75, 100, 150]

# =============================================================================
# AUTOENCODER HYPERPARAMETERS
# Neural network architecture and training configuration
# =============================================================================
AE_HIDDEN_LAYERS = [256, 128]  # Hidden layer sizes for encoder/decoder
AE_EPOCHS = 200  # Maximum training epochs
AE_BATCH_SIZE = 64  # Mini-batch size for training
AE_LEARNING_RATE = 1e-3  # Adam optimizer learning rate
AE_WEIGHT_DECAY = 1e-4  # L2 regularization coefficient
AE_PATIENCE = 10  # Early stopping patience (epochs without improvement)
AE_DROPOUT_RATE = 0.1  # Dropout probability for regularization

# =============================================================================
# CROSS-VALIDATION SETTINGS
# Configuration for model selection and hyperparameter tuning
# =============================================================================
CV_FOLDS = 5  # Number of folds for stratified cross-validation