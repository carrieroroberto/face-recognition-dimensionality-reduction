# config.py
import os

# General
RANDOM_STATE = 42
VERBOSE = True

# Dataset
MIN_FACES_PER_PERSON = 60
RESIZE_FACTOR = 0.75
TEST_SIZE = 0.20

# PCA
COMPONENTS_RANGE = [25, 50, 75, 100, 125, 150, 175, 200]
COMPONENTS_FINAL = 150

# Autoencoder
AE_HIDDEN_LAYERS = [512, 256]
AE_DROPOUT = 0.1
AE_EPOCHS = 200
AE_BATCH_SIZE = 32
AE_LEARNING_RATE = 1e-3
AE_WEIGHT_DECAY = 1e-5
AE_PATIENCE = 25

# SVM
SVM_CV_FOLDS = 5
SVM_PARAM_GRID = {
    'C': [1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale']
}

# Neural Network
NN_HIDDEN_LAYERS = (256, 128)
NN_DROPOUT = 0.2
NN_PARAM_GRID = {
    'learning_rate': [1e-3, 5e-4],
    'weight_decay': [1e-5]
}
NN_EPOCHS = 300
NN_BATCH_SIZE = 32
NN_PATIENCE = 40
NN_USE_CLASS_WEIGHTS = True
NN_LR_SCHEDULER = True
NN_LR_PATIENCE = 10
NN_LR_FACTOR = 0.5

# Cross-validation
CV_N_FOLDS = 5
CV_N_JOBS = -1

# Verification
VERIFICATION_N_PAIRS = 1000
VERIFICATION_METRICS = ['cosine', 'euclidean']

# Plot style
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
PLOT_DPI = 150
PLOT_FIGSIZE_SMALL = (8, 6)
PLOT_FIGSIZE_MEDIUM = (10, 8)
PLOT_FIGSIZE_LARGE = (14, 10)
PLOT_COLORMAP = 'viridis'
N_EIGENFACES_DISPLAY = 16

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "results", "models")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "figures")
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics")