# src/pca.py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import config
from src.classification import train_svm
from src.utils import plot_scree_plot, plot_eigenfaces


class SVD_PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        # 1. Centratura dei dati
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Calcolo della SVD economica (full_matrices=False)
        # U: coordinate nel nuovo spazio, S: valori singolari, Vt: componenti
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # 3. Selezione delle prime k componenti
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]

        # 4. Calcolo della varianza spiegata (per lo Scree Plot)
        # La varianza è proporzionale al quadrato dei valori singolari
        total_variance = np.sum(S ** 2)
        self.explained_variance_ratio_ = (S[:self.n_components] ** 2) / total_variance

        return self

    def transform(self, X):
        # Proiezione nello spazio ridotto: (X - mean) * V
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_reduced):
        # Ricostruzione dell'immagine originale: (X_red * V^T) + mean
        return np.dot(X_reduced, self.components_) + self.mean_

def run_pca_experiments(X_train, X_test, y_train, y_test, h, w, n_components_list):
    """Testa la PCA con diversi numeri di componenti principali e salva grafici di accuracy e varianza."""
    results = {}
    explained_variances = []

    for n_comp in n_components_list:
        print(f"\n--- PCA con {n_comp} componenti ---")
        pca = SVD_PCA(n_components=n_comp)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        X_test_pca_rec = pca.inverse_transform(X_test_pca)

        # Salva varianza cumulativa
        explained_variances.append(np.sum(pca.explained_variance_ratio_))

        # Grafici
        plot_eigenfaces(pca, h, w, n_top=min(12, n_comp))
        plot_scree_plot(pca)

        # Standardizzazione
        scaler = StandardScaler().fit(X_train_pca)
        X_train_scaled = scaler.transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)

        # Classificazione SVM
        svm_model = train_svm(X_train_scaled, y_train)
        y_pred = svm_model.predict(X_test_scaled)
        acc = np.round(np.mean(y_pred == y_test), 4)
        print(f"Accuratezza SVM: {acc}")

        results[n_comp] = {
            "pca_model": pca,
            "scaler": scaler,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "svm_acc": acc
        }

    # Plot accuracies vs numero componenti
    plt.figure(figsize=(8, 5))
    plt.plot(list(results.keys()), [results[n]['svm_acc'] for n in results], marker='o', label='Accuracy')
    plt.plot(n_components_list, explained_variances, marker='x', label='Varianza spiegata cumulativa')
    plt.xlabel("Numero componenti principali")
    plt.ylabel("Accuracy / Varianza spiegata")
    plt.title("Accuratezza e Varianza vs Numero Componenti PCA")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{config.OUTPUT_PATH}/accuracy_varianza_vs_pca_components.png", bbox_inches='tight')
    plt.close()

    return results