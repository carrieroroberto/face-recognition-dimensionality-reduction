# src/utils.py
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import config

def load_and_preprocess_data():
    """Carica il dataset LFW, normalizza e divide in Train/Test."""
    print("Caricamento dataset LFW...")

    # Carichiamo il dataset
    lfw_people = fetch_lfw_people(
        min_faces_per_person=config.MIN_FACES_PER_PERSON,
        resize=0.4,  # Ridimensiona per mantenere IMAGE_SHAPE (50x37)
        color=False
    )

    # Informazioni dataset
    X = lfw_people.data
    y = lfw_people.target
    target_names = lfw_people.target_names
    images = lfw_people.images
    n_samples, h, w = images.shape

    # Normalizzazione [0, 1]
    X = X / 255.0

    # Split Train/Test stratificato (mantiene le proporzioni delle classi)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE
    )

    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'target_names': target_names,
        'image_shape': (h, w),
        'images': images
    }

def plot_dataset_stats(data_dict):
    """Genera output grafici per l'EDA (Exploratory Data Analysis)."""
    target_names = data_dict['target_names']
    y_train = data_dict['y_train']
    h, w = data_dict['image_shape']
    X_train = data_dict['X_train']

    # 1. Distribuzione Classi
    plt.figure(figsize=(10, 5))
    unique, counts = np.unique(y_train, return_counts=True)
    sns.barplot(x=[target_names[i] for i in unique], y=counts, palette='viridis')
    plt.title("Distribuzione delle Classi (Train Set)")
    plt.xticks(rotation=45)
    plt.ylabel("Numero di Immagini")
    plt.savefig(f"{config.OUTPUT_PATH}/class_distribution.png", bbox_inches='tight')
    plt.close()

    # 2. La "Faccia Media" (Concetto matematico importante per PCA)
    plt.figure(figsize=(4, 4))
    mean_face = np.mean(X_train, axis=0).reshape(h, w)
    plt.imshow(mean_face, cmap='gray')
    plt.title("Faccia Media del Dataset")
    plt.axis('off')
    plt.savefig(f"{config.OUTPUT_PATH}/mean_face.png", bbox_inches='tight')
    plt.close()

    print(f"Grafici salvati in {config.OUTPUT_PATH}")

def plot_eigenfaces(pca_model, h, w, n_top=12):
    """Visualizza le prime n componenti principali come immagini."""
    plt.figure(figsize=(12, 8))
    for i in range(n_top):
        plt.subplot(3, 4, i + 1)
        # Ogni riga di Vt è una "faccia fantasma"
        eigenface = pca_model.components_[i].reshape(h, w)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f"Eigenface {i+1}")
        plt.axis('off')
    plt.suptitle("Prime Componenti Principali (SVD)")
    plt.savefig(f"{config.OUTPUT_PATH}/eigenfaces.png", bbox_inches='tight')
    plt.close()

def plot_scree_plot(pca_model):
    """Grafico della varianza spiegata cumulativa."""
    plt.figure(figsize=(8, 5))
    cum_variance = np.cumsum(pca_model.explained_variance_ratio_)
    plt.plot(range(1, len(cum_variance) + 1), cum_variance, marker='o', linestyle='--')
    plt.xlabel("Numero di Componenti")
    plt.ylabel("Varianza Spiegata Cumulativa")
    plt.title("Analisi della Varianza (Scree Plot)")
    plt.grid(True)
    plt.savefig(f"{config.OUTPUT_PATH}/scree_plot.png", bbox_inches='tight')
    plt.close()

def plot_training_loss(loss_history):
    """Visualizza l'andamento della loss dell'Autoencoder."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='tab:orange', lw=2)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoca")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig(f"{config.OUTPUT_PATH}/ae_loss.png", bbox_inches='tight')
    plt.close()

def plot_reconstruction_comparison(original, pca_rec, ae_rec, h, w, n_images=5):
    """
    Mostra un confronto visivo tra immagini originali e ricostruite.
    original, pca_rec, ae_rec: matrici (N, pixels)
    """
    plt.figure(figsize=(15, 3 * n_images))

    for i in range(n_images):
        # Originale
        plt.subplot(n_images, 3, i * 3 + 1)
        plt.imshow(original[i].reshape(h, w), cmap='gray')
        plt.title("Originale")
        plt.axis('off')

        # Ricostruzione PCA
        plt.subplot(n_images, 3, i * 3 + 2)
        plt.imshow(pca_rec[i].reshape(h, w), cmap='gray')
        plt.title(f"PCA (SVD k={config.N_COMPONENTS_PCA})")
        plt.axis('off')

        # Ricostruzione Autoencoder
        plt.subplot(n_images, 3, i * 3 + 3)
        plt.imshow(ae_rec[i].reshape(h, w), cmap='gray')
        plt.title(f"Autoencoder (Latent={config.LATENT_DIM_AE})")
        plt.axis('off')

    plt.savefig(f"{config.OUTPUT_PATH}/reconstruction_comparison.png", bbox_inches='tight')
    plt.close()
    print(f"Confronto ricostruzioni salvato in {config.OUTPUT_PATH}")

def plot_confusion_matrix(y_true, y_pred, target_names, model_name):
    """Visualizza la matrice di confusione con heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel('Classe Reale')
    plt.xlabel('Classe Predetta')
    plt.savefig(f"{config.OUTPUT_PATH}/cm_{model_name.lower().replace(' ', '_')}.png", bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_test, y_score, n_classes, model_name):
    """Genera le curve ROC per ogni classe (One-vs-Rest)."""
    # Binarizzazione per multi-classe
    y_test_bin = label_binarize(y_test, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Classe {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves: {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"{config.OUTPUT_PATH}/roc_{model_name.lower().replace(' ', '_')}.png", bbox_inches='tight')
    plt.close()

def plot_tsne_comparison(pca_features, ae_features, labels, target_names):
    """Visualizza i cluster di PCA vs Autoencoder in 2D."""
    tsne = TSNE(n_components=2, random_state=config.RANDOM_STATE)

    plt.figure(figsize=(16, 7))

    for i, (name, feat) in enumerate([("PCA", pca_features), ("Autoencoder", ae_features)]):
        X_2d = tsne.fit_transform(feat)
        plt.subplot(1, 2, i + 1)
        for g in np.unique(labels):
            idx = np.where(labels == g)
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=target_names[g], alpha=0.6)
        plt.title(f"t-SNE Projection: {name} Features")
        if i == 1: plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(f"{config.OUTPUT_PATH}/tsne_comparison.png", bbox_inches='tight')
    plt.close()

def save_model(model, name):
    """Salva il modello addestrato."""
    if isinstance(model, torch.nn.Module):
        path = os.path.join(config.MODELS_PATH, f"{name}.pt")
        torch.save(model.state_dict(), path)
    else:
        path = os.path.join(config.MODELS_PATH, f"{name}.joblib")
        joblib.dump(model, path)
    print(f"Modello salvato: {path}")