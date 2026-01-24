import os
import numpy as np
import torch
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, "results", "models")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "figures")

MIN_FACES_PER_PERSON = 70 
RESIZE_RATIO = 0.5
TEST_SIZE = 0.2

COMPONENTS_RANGE = [25, 50, 75, 100, 150]

AE_HIDDEN_LAYERS = [256, 128]
AE_EPOCHS = 200
AE_BATCH_SIZE = 64
AE_LEARNING_RATE = 1e-3
AE_WEIGHT_DECAY = 1e-4
AE_PATIENCE = 10
AE_DROPOUT_RATE = 0.1

CV_FOLDS = 5