# config.py
import os

# Parametri Dataset
MIN_FACES_PER_PERSON = 50
IMAGE_SHAPE = (64, 64)  # Ridimensionamento per efficienza
RANDOM_STATE = 42

# Parametri Riduzione Dimensionale
N_COMPONENTS_PCA = 150
LATENT_DIM_AE = 150

# Path
DATA_PATH = "./data"
MODELS_PATH = "./models"
OUTPUT_PATH = "./output"

# Creazione cartelle se non esistono
for path in [DATA_PATH, MODELS_PATH, OUTPUT_PATH]:
    os.makedirs(path, exist_ok=True)