"""
This module implements Principal Component Analysis (PCA) from scratch using
Singular Value Decomposition (SVD). PCA projects high-dimensional data onto
a lower-dimensional subspace that captures the maximum variance, making it
effective for face recognition (known as Eigenfaces method).

The implementation follows the mathematical formulation:
X_centered = X - mean(X)
U, S, V^T = SVD(X_centered)
X_reduced = X_centered @ V[:k].T
"""

import numpy as np
from sklearn.svm import SVC
from src.utils import plot_scree_plot, plot_eigenfaces, plot_ablation_study


class SVD_PCA:
    """
    Principal Component Analysis implementation using Singular Value Decomposition.

    This class provides methods to fit PCA on training data and transform
    new data to the learned lower-dimensional representation.

    Attributes:
        n_components: Number of principal components to retain
        components_: Principal component vectors (eigenvectors of covariance)
        mean_: Mean of training data (for centering)
        singular_values_: Singular values corresponding to each component
        explained_variance_ratio_: Proportion of variance explained by each component
    """

    def __init__(self, n_components):
        """
        Initialize PCA with the desired number of components.

        Args:
            n_components: Number of principal components to retain
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.singular_values_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        Fit PCA model to training data using SVD.

        Centers the data by subtracting the mean, then performs SVD to
        extract the principal components (right singular vectors).

        Args:
            X: Training data array of shape (n_samples, n_features)

        Returns:
            self: The fitted PCA model
        """
        # Center data by subtracting mean
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Perform Singular Value Decomposition
        # U: left singular vectors, S: singular values, Vt: right singular vectors
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Store top-k principal components (rows of Vt)
        self.components_ = Vt[:self.n_components]
        self.singular_values_ = S[:self.n_components]

        # Compute explained variance ratio from singular values
        total_variance = np.sum(S ** 2)
        self.explained_variance_ratio_ = (S[:self.n_components] ** 2) / total_variance

        return self

    def transform(self, X):
        """
        Project data onto the principal component subspace.

        Args:
            X: Data array of shape (n_samples, n_features)

        Returns:
            numpy.ndarray: Transformed data of shape (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_reduced):
        """
        Reconstruct original data from reduced representation.

        Projects the reduced data back to the original feature space.
        Note: Information is lost due to dimensionality reduction.

        Args:
            X_reduced: Reduced data of shape (n_samples, n_components)

        Returns:
            numpy.ndarray: Reconstructed data of shape (n_samples, n_features)
        """
        return np.dot(X_reduced, self.components_) + self.mean_


def run_pca_experiments(n_components_list, X_train, X_test, y_train, y_test, h, w, verbose=True):
    """
    Run ablation study over different numbers of principal components.

    Trains PCA models with varying component counts and evaluates their
    performance using SVM classification on the projected features.

    Args:
        n_components_list: List of component counts to test
        X_train: Training features array
        X_test: Test features array
        y_train: Training labels
        y_test: Test labels
        h: Image height (for eigenface visualization)
        w: Image width (for eigenface visualization)
        verbose: Whether to print progress

    Returns:
        dict: Results for each component count containing PCA model, features,
              variance explained, reconstruction error, and SVM accuracy
    """
    results = {}
    explained_variances = []
    accuracies = []

    # Iterate over each component count configuration
    for n_comp in n_components_list:
        if verbose:
            print(f"\n--- PCA with {n_comp} components ---")

        # Fit PCA model on training data
        pca = SVD_PCA(n_components=n_comp)
        pca.fit(X_train)

        # Transform both training and test data
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Compute reconstruction quality
        X_test_pca_rec = pca.inverse_transform(X_test_pca)
        reconstruction_mse = np.mean((X_test - X_test_pca_rec) ** 2)

        # Track cumulative variance explained
        variance_explained = np.sum(pca.explained_variance_ratio_)
        explained_variances.append(variance_explained)

        # Generate visualizations
        plot_eigenfaces(pca, h, w, n_top=min(12, n_comp))
        plot_scree_plot(pca)

        # Evaluate feature quality using SVM classifier
        svm_model = SVC(kernel="rbf")
        svm_model.fit(X_train_pca, y_train)

        # Compute test accuracy
        y_pred = svm_model.predict(X_test_pca)
        acc = np.mean(y_pred == y_test)
        accuracies.append(acc)

        if verbose:
            print(f"   Variance explained: {variance_explained*100:.2f}%")
            print(f"   Reconstruction MSE: {reconstruction_mse:.6f}")
            print(f"   SVM Test Accuracy: {acc:.4f}")

        # Store results for this configuration
        results[n_comp] = {
            "pca_model": pca,
            "X_train_pca": X_train_pca,
            "X_test_pca": X_test_pca,
            "variance_explained": variance_explained,
            "reconstruction_mse": reconstruction_mse,
            "svm_model": svm_model,
            "svm_test_acc": acc
        }

    # Generate ablation study visualization
    plot_ablation_study(
        x_values=n_components_list,
        accuracy_scores=accuracies,
        secondary_metric_scores=explained_variances,
        x_label="Number of Principal Components",
        secondary_label="Cumulative Variance Explained",
        title="Ablation Study: PCA Components vs Performance",
    )

    return results