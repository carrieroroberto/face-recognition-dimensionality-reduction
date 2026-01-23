# src/verification.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_curve, auc
import config


def generate_pairs(features, labels, n_pairs=1000):
    """
    Genera coppie bilanciate (50% stessa persona, 50% persone diverse)
    per il test di verifica.
    """
    np.random.seed(config.RANDOM_STATE)
    n_samples = features.shape[0]
    pair_features_1 = []
    pair_features_2 = []
    pair_labels = []  # 1 se stessa persona, 0 altrimenti

    # Generazione coppie positive (Stessa persona)
    unique_labels = np.unique(labels)
    while len(pair_labels) < n_pairs // 2:
        label = np.random.choice(unique_labels)
        idx = np.where(labels == label)[0]
        if len(idx) >= 2:
            i, j = np.random.choice(idx, 2, replace=False)
            pair_features_1.append(features[i])
            pair_features_2.append(features[j])
            pair_labels.append(1)

    # Generazione coppie negative (Persone diverse)
    while len(pair_labels) < n_pairs:
        i, j = np.random.choice(n_samples, 2, replace=False)
        if labels[i] != labels[j]:
            pair_features_1.append(features[i])
            pair_features_2.append(features[j])
            pair_labels.append(0)

    return np.array(pair_features_1), np.array(pair_features_2), np.array(pair_labels)


def calculate_eer(fpr, tpr, thresholds):
    """Calcola l'Equal Error Rate e la soglia ottimale."""
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[idx]
    threshold = thresholds[idx]
    return eer, threshold


def run_verification_study(pca_features, ae_features, labels):
    """
    Esegue lo studio comparativo richiesto:
    PCA vs AE e Cosine vs Euclidean.
    """
    metrics = ['cosine', 'euclidean']
    feature_sets = {'PCA': pca_features, 'Autoencoder': ae_features}

    plt.figure(figsize=(12, 8))

    results = {}

    for feat_name, feat_data in feature_sets.items():
        f1, f2, y_true = generate_pairs(feat_data, labels)

        for metric in metrics:
            if metric == 'cosine':
                # Similarità: più alto è, meglio è
                scores = np.array(
                    [cosine_similarity(f1[i].reshape(1, -1), f2[i].reshape(1, -1))[0][0] for i in range(len(y_true))])
            else:
                # Distanza: più basso è, meglio è (usiamo il negativo per la ROC)
                dists = np.array(
                    [euclidean_distances(f1[i].reshape(1, -1), f2[i].reshape(1, -1))[0][0] for i in range(len(y_true))])
                scores = -dists  # Invertiamo perché la ROC si aspetta 'maggior score = più probabile classe 1'

            fpr, tpr, thresholds = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            eer, best_th = calculate_eer(fpr, tpr, thresholds)

            label_str = f"{feat_name} + {metric.capitalize()} (AUC: {roc_auc:.3f}, EER: {eer:.3f})"
            plt.plot(fpr, tpr, label=label_str)

            results[f"{feat_name}_{metric}"] = {'auc': roc_auc, 'eer': eer, 'threshold': best_th}

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Acceptance Rate (FAR)')
    plt.ylabel('True Acceptance Rate (TAR)')
    plt.title('Face Verification: Comparative Ablation Study')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{config.OUTPUT_PATH}/verification_ablation_study.png")

    return results