# src/utils.py
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import config


def _setup_style():
    plt.style.use('seaborn-v0_8-whitegrid')


def plot_dataset_stats(data_dict):
    _setup_style()
    target_names = data_dict['target_names']
    y_train = data_dict['y_train']
    h, w = data_dict['image_shape']
    X_train = data_dict['X_train']

    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE_MEDIUM)
    unique, counts = np.unique(y_train, return_counts=True)
    sns.barplot(x=[target_names[i] for i in unique], y=counts, palette='viridis', ax=ax)
    ax.set_title("Distribuzione delle Classi (Train Set)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Numero di Immagini", fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/class_distribution.png", dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    mean_face = np.mean(X_train, axis=0).reshape(h, w)
    ax.imshow(mean_face, cmap='gray')
    ax.set_title("Faccia Media del Dataset", fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/mean_face.png", dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    print(f"  Grafici salvati in {config.OUTPUT_PATH}")


def plot_eigenfaces(pca_model, h, w, n_top=12):
    _setup_style()
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for i in range(min(n_top, len(pca_model.components_))):
        eigenface = pca_model.components_[i].reshape(h, w)
        axes[i].imshow(eigenface, cmap='gray')
        axes[i].set_title(f"Eigenface {i+1}", fontsize=10)
        axes[i].axis('off')

    plt.suptitle("Prime Componenti Principali (SVD)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/eigenfaces.png", dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_scree_plot(pca_model):
    _setup_style()
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE_SMALL)
    cum_variance = np.cumsum(pca_model.explained_variance_ratio_)
    ax.plot(range(1, len(cum_variance) + 1), cum_variance, marker='o', linestyle='--', markersize=4)
    ax.set_xlabel("Numero di Componenti", fontsize=11)
    ax.set_ylabel("Varianza Spiegata Cumulativa", fontsize=11)
    ax.set_title("Analisi della Varianza (Scree Plot)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/scree_plot.png", dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_training_loss(loss_history):
    _setup_style()
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE_SMALL)
    ax.plot(loss_history, color='tab:orange', lw=2)
    ax.set_title("Autoencoder Training Loss", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoca", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/ae_loss.png", dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_reconstruction_comparison(original, pca_rec, ae_rec, h, w, n_images=5):
    _setup_style()
    fig, axes = plt.subplots(n_images, 3, figsize=(10, 3 * n_images))

    for i in range(n_images):
        axes[i, 0].imshow(original[i].reshape(h, w), cmap='gray')
        axes[i, 0].set_title("Originale" if i == 0 else "", fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pca_rec[i].reshape(h, w), cmap='gray')
        axes[i, 1].set_title(f"PCA" if i == 0 else "", fontsize=10)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(ae_rec[i].reshape(h, w), cmap='gray')
        axes[i, 2].set_title(f"Autoencoder" if i == 0 else "", fontsize=10)
        axes[i, 2].axis('off')

    plt.suptitle("Confronto Ricostruzioni", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/reconstruction_comparison.png", dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Confronto ricostruzioni salvato")


def plot_confusion_matrix(y_true, y_pred, target_names, model_name):
    _setup_style()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE_MEDIUM)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight='bold')
    ax.set_ylabel('Classe Reale', fontsize=11)
    ax.set_xlabel('Classe Predetta', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/cm_{model_name.lower().replace(' ', '_')}.png",
                dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()


def plot_tsne_comparison(pca_features, ae_features, labels, target_names):
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=config.PLOT_FIGSIZE_LARGE)

    for i, (name, feat) in enumerate([("PCA", pca_features), ("Autoencoder", ae_features)]):
        tsne = TSNE(n_components=2, random_state=config.RANDOM_STATE, perplexity=30)
        X_2d = tsne.fit_transform(feat)

        for g in np.unique(labels):
            idx = np.where(labels == g)
            axes[i].scatter(X_2d[idx, 0], X_2d[idx, 1], label=target_names[g], alpha=0.6, s=20)

        axes[i].set_title(f"t-SNE: {name} Features", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("t-SNE 1", fontsize=10)
        axes[i].set_ylabel("t-SNE 2", fontsize=10)

    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/tsne_comparison.png", dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()


def save_model(model, name):
    if isinstance(model, torch.nn.Module):
        path = os.path.join(config.MODELS_PATH, f"{name}.pt")
        torch.save(model.state_dict(), path)
    else:
        path = os.path.join(config.MODELS_PATH, f"{name}.joblib")
        joblib.dump(model, path)
    print(f"  Modello salvato: {path}")
