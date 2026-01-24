import numpy as np
from sklearn.svm import SVC
from src.utils import plot_scree_plot, plot_eigenfaces, plot_ablation_study

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


def run_pca_experiments(n_components_list, X_train, X_test, y_train, y_test, h, w, verbose=True):
    results = {}
    explained_variances = []
    accuracies = []

    for n_comp in n_components_list:
        if verbose:
            print(f"\n--- PCA with {n_comp} components ---")

        pca = SVD_PCA(n_components=n_comp)
        pca.fit(X_train)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        X_test_pca_rec = pca.inverse_transform(X_test_pca)
        reconstruction_mse = np.mean((X_test - X_test_pca_rec) ** 2)

        variance_explained = np.sum(pca.explained_variance_ratio_)
        explained_variances.append(variance_explained)

        plot_eigenfaces(pca, h, w, n_top=min(12, n_comp))
        plot_scree_plot(pca)

        svm_model = SVC(kernel='rbf')
        svm_model.fit(X_train_pca, y_train)

        y_pred = svm_model.predict(X_test_pca)
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc)

        if verbose:
            print(f"   Variance explained: {variance_explained*100:.2f}%")
            print(f"   Reconstruction MSE: {reconstruction_mse:.6f}")
            print(f"   SVM Test Accuracy: {acc:.4f}")

        results[n_comp] = {
            "pca_model": pca,
            "X_train_pca": X_train_pca,
            "X_test_pca": X_test_pca,
            "variance_explained": variance_explained,
            "reconstruction_mse": reconstruction_mse,
            "svm_model": svm_model,
            "svm_test_acc": acc
        }

    plot_ablation_study(
        x_values=n_components_list,
        accuracy_scores=accuracies,
        secondary_metric_scores=explained_variances,
        x_label="Number of Principal Components",
        secondary_label="Cumulative Variance Explained",
        title="Ablation Study: PCA Components vs Performance",
    )
    
    return results