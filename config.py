# config.py
import os

RANDOM_STATE = 42
VERBOSE = True

MIN_FACES_PER_PERSON = 40
RESIZE_FACTOR = 0.4
TEST_SIZE = 0.25

PCA_COMPONENTS_RANGE = [10, 25, 50, 75, 100, 125, 150, 175, 200]
N_COMPONENTS_PCA_FINAL = 150
N_COMPONENTS_PCA = N_COMPONENTS_PCA_FINAL

AE_LATENT_DIM_RANGE = [10, 25, 50, 75, 100, 125, 150, 175, 200]
LATENT_DIM_AE_FINAL = 150
LATENT_DIM_AE = LATENT_DIM_AE_FINAL

AE_HIDDEN_LAYERS = [512, 256]
AE_ACTIVATION = 'leaky_relu'
AE_USE_BATCH_NORM = True
AE_DROPOUT_RATE = 0.2

AE_EPOCHS = 200
AE_BATCH_SIZE = 32
AE_LEARNING_RATE = 1e-3
AE_WEIGHT_DECAY = 1e-5
AE_PATIENCE = 15

SVM_CV_FOLDS = 5
SVM_PARAM_GRID = {
    'C': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

NN_PARAM_GRID = {
    'hidden_layers': [(128, 64)],
    'learning_rate': [1e-2, 1e-3],
    'dropout_rate': [0.2, 0.3],
    'weight_decay': [1e-5, 1e-4]
}

NN_EPOCHS = 200
NN_BATCH_SIZE = 32
NN_PATIENCE = 15
NN_LR_SCHEDULER = True
NN_LR_PATIENCE = 10
NN_LR_FACTOR = 0.5
CV_N_FOLDS = 5
CV_N_JOBS = -1

VERIFICATION_N_PAIRS = 1000
VERIFICATION_METRICS = ['cosine', 'euclidean']

PLOT_STYLE = 'seaborn-v0_8-whitegrid'
PLOT_DPI = 150
PLOT_FIGSIZE_SMALL = (8, 6)
PLOT_FIGSIZE_MEDIUM = (12, 8)
PLOT_FIGSIZE_LARGE = (16, 10)
PLOT_COLORMAP = 'viridis'

N_EIGENFACES_DISPLAY = 16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data")
MODELS_PATH = os.path.join(BASE_DIR, "results", "models")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "figures")
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics")

for path in [DATA_PATH, MODELS_PATH, OUTPUT_PATH, METRICS_PATH]:
    os.makedirs(path, exist_ok=True)

LOG_FILE = os.path.join(BASE_DIR, "results", "experiment.log")

def get_config_summary():
    return {
        'Dataset': {
            'MIN_FACES_PER_PERSON': MIN_FACES_PER_PERSON,
            'RESIZE_FACTOR': RESIZE_FACTOR,
            'TEST_SIZE': TEST_SIZE
        },
        'PCA': {
            'Components Range': PCA_COMPONENTS_RANGE,
            'Final Components': N_COMPONENTS_PCA_FINAL
        },
        'Autoencoder': {
            'Latent Dim Range': AE_LATENT_DIM_RANGE,
            'Final Latent Dim': LATENT_DIM_AE_FINAL,
            'Architecture': f"Input -> {AE_HIDDEN_LAYERS} -> Latent",
            'Epochs': AE_EPOCHS,
            'Batch Size': AE_BATCH_SIZE,
            'Learning Rate': AE_LEARNING_RATE
        },
        'Training': {
            'CV Folds': CV_N_FOLDS,
            'Random State': RANDOM_STATE
        }
    }

def print_config():
    print("CONFIGURAZIONE PROGETTO")
    summary = get_config_summary()
    for section, params in summary.items():
        print(f"\n{section}:")
        for key, value in params.items():
            print(f"  {key}: {value}")