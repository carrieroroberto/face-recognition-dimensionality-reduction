# src/pca.py
import numpy as np
from matplotlib import pyplot as plt
import config
from src.classification import train_svm_with_grid_search
from src.utils import plot_scree_plot, plot_eigenfaces

class SVD_PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]

        total_variance = np.sum(S ** 2)
        self.explained_variance_ratio_ = (S[:self.n_components] ** 2) / total_variance

        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_reduced):
        return np.dot(X_reduced, self.components_) + self.mean_


def run_pca_experiments(X_train, X_test, y_train, y_test, h, w, n_components_list,
                        use_grid_search=False, verbose=True):
    results = {}
    explained_variances = []
    accuracies = []

    for n_comp in n_components_list:
        if verbose:
            print(f"\n--- PCA con {n_comp} componenti ---")

        pca = SVD_PCA(n_components=n_comp)
        pca.fit(X_train)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        X_test_pca_rec = pca.inverse_transform(X_test_pca)
        reconstruction_mse = np.mean((X_test - X_test_pca_rec) ** 2)

        variance_explained = np.sum(pca.explained_variance_ratio_)
        explained_variances.append(variance_explained)

        if n_comp in [10, 50, 100, 150]:
            plot_eigenfaces(pca, h, w, n_top=min(12, n_comp))
            plot_scree_plot(pca)

        if use_grid_search:
            svm_model, svm_cv_score, svm_params = train_svm_with_grid_search(
                X_train_pca, y_train, verbose=False
            )
        else:
            from sklearn.svm import SVC
            svm_model = SVC(kernel='linear', C=1.0, random_state=config.RANDOM_STATE)
            svm_model.fit(X_train_pca, y_train)
            svm_cv_score = None
            svm_params = {'kernel': 'linear', 'C': 1.0}

        y_pred = svm_model.predict(X_test_pca)
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc)

        if verbose:
            print(f"  Varianza spiegata: {variance_explained*100:.2f}%")
            print(f"  Reconstruction MSE: {reconstruction_mse:.6f}")
            if svm_cv_score:
                print(f"  SVM CV Score: {svm_cv_score:.4f}")
            print(f"  SVM Test Accuracy: {acc:.4f}")

        results[n_comp] = {
            "pca_model": pca,
            "X_train_pca": X_train_pca,
            "X_test_pca": X_test_pca,
            "variance_explained": variance_explained,
            "reconstruction_mse": reconstruction_mse,
            "svm_model": svm_model,
            "svm_cv_score": svm_cv_score,
            "svm_params": svm_params,
            "svm_test_acc": acc
        }

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=config.PLOT_FIGSIZE_MEDIUM)

    ax1 = plt.gca()
    ax1.set_xlabel("Numero Componenti Principali")
    ax1.set_ylabel("Test Accuracy", color='tab:blue')
    ax1.plot(n_components_list, accuracies, marker='o', color='tab:blue',
             linewidth=2, markersize=8, label='SVM Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Varianza Spiegata Cumulativa", color='tab:orange')
    ax2.plot(n_components_list, explained_variances, marker='s', color='tab:orange',
             linewidth=2, markersize=8, linestyle='--', label='Varianza')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title("Ablation Study: PCA Components vs Performance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/pca_ablation_study.png",
                dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"\nGrafico ablation study salvato in {config.OUTPUT_PATH}")

    return results