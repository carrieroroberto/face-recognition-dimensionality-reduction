"""
This module provides two classifier implementations for multi-class face recognition:
1. Support Vector Machine (SVM) with RBF/linear kernels
2. Neural Network classifier with configurable architecture

Both classifiers support hyperparameter tuning via grid search with stratified
cross-validation to ensure robust model selection.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import config


def train_svm_with_grid_search(X_train, y_train, param_grid, cv_folds=config.CV_FOLDS):
    """
    Train an SVM classifier with hyperparameter optimization via grid search.

    Performs exhaustive search over specified parameter combinations using
    stratified k-fold cross-validation to find the best hyperparameters.

    Args:
        X_train: Training features array of shape (n_samples, n_features)
        y_train: Training labels array of shape (n_samples,)
        param_grid: Dictionary with parameter names as keys and lists of values to try
        cv_folds: Number of cross-validation folds

    Returns:
        tuple: (best_estimator, best_cv_score, best_params) containing the
               fitted model with optimal parameters and validation performance
    """
    print("\n--- SVM GRID SEARCH STARTING ---")
    print(f"Hyperparameters grid: {param_grid}")
    print(f"Cross-Validation: {cv_folds} folds")

    # Initialize stratified k-fold cross-validator
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True)

    # Configure SVM with probability estimates and class balancing
    svm = SVC(probability=True, class_weight="balanced")

    # Setup grid search with parallel processing
    grid = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=True
    )

    # Execute grid search
    grid.fit(X_train, y_train)

    print("\n--- SVM GRID SEARCH COMPLETED ---")
    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV Score: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_score_, grid.best_params_


class NeuralNetwork(BaseEstimator, ClassifierMixin):
    """
    Multi-layer perceptron classifier compatible with scikit-learn API.

    Implements a feedforward neural network with configurable hidden layers,
    batch normalization, dropout regularization, and class-weighted loss
    for handling imbalanced datasets.

    Attributes:
        hidden_layers: Tuple of hidden layer sizes
        epochs: Number of training epochs
        lr: Learning rate for Adam optimizer
        weight_decay: L2 regularization coefficient
        dropout_rate: Dropout probability between layers
        batch_size: Mini-batch size for training
        model_: The underlying PyTorch model
        classes_: Array of unique class labels
    """

    def __init__(self, hidden_layers=(64,), epochs=200, lr=0.01,
                 weight_decay=0.001, dropout_rate=0.2, batch_size=32, verbose=True):
        """
        Initialize neural network hyperparameters.

        Args:
            hidden_layers: Tuple specifying the size of each hidden layer
            epochs: Maximum number of training iterations
            lr: Learning rate for the Adam optimizer
            weight_decay: L2 regularization strength
            dropout_rate: Probability of dropping units during training
            batch_size: Number of samples per gradient update
            verbose: Whether to print training progress
        """
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.verbose = verbose

        self.model_ = None
        self.classes_ = None
        self.history_ = {}

    def _build_model(self, input_dim, n_classes):
        """
        Construct the neural network architecture.

        Creates a sequential model with:
        - Linear layers followed by batch normalization
        - ReLU activation functions
        - Dropout for regularization
        - Final linear layer for class logits

        Args:
            input_dim: Number of input features
            n_classes: Number of output classes

        Returns:
            nn.Sequential: The constructed PyTorch model
        """
        layers = []
        prev_dim = input_dim

        # Build hidden layers with normalization and activation
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim

        # Output layer for classification
        layers.append(nn.Linear(prev_dim, n_classes))
        model = nn.Sequential(*layers)

        # Initialize weights using Xavier normal initialization
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        return model

    def fit(self, X, y):
        """
        Train the neural network on the provided data.

        Uses cross-entropy loss with class weights to handle imbalanced data
        and Adam optimizer for gradient descent.

        Args:
            X: Training features array of shape (n_samples, n_features)
            y: Training labels array of shape (n_samples,)

        Returns:
            self: The fitted classifier
        """
        # Store class information
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build the model architecture
        self.model_ = self._build_model(n_features, n_classes)

        # Compute class weights for imbalanced data handling
        class_counts = np.bincount(y)
        safe_counts = np.maximum(class_counts, 1)  # Avoid division by zero
        weights = 1.0 / safe_counts
        weights = weights / weights.sum() * n_classes  # Normalize weights
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))

        # Initialize optimizer
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Training history tracking
        self.history_ = {"loss": [], "acc": []}
        self.model_.train()

        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate metrics
                total_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

            # Record epoch metrics
            avg_loss = total_loss / len(dataset)
            acc = correct / total
            self.history_["loss"].append(avg_loss)
            self.history_["acc"].append(acc)

            # Print progress at regular intervals
            if self.verbose and (epoch == 0 or (epoch + 1) % 20 == 0):
                print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

        return self

    def predict(self, X):
        """
        Predict class labels for input samples.

        Args:
            X: Input features array of shape (n_samples, n_features)

        Returns:
            numpy.ndarray: Predicted class labels of shape (n_samples,)
        """
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(torch.FloatTensor(X))
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Applies softmax to the model outputs to obtain probability distributions.

        Args:
            X: Input features array of shape (n_samples, n_features)

        Returns:
            numpy.ndarray: Class probabilities of shape (n_samples, n_classes)
        """
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(torch.FloatTensor(X))
            probs = torch.softmax(outputs, dim=1)
        return probs.numpy()


def train_nn_with_grid_search(X_train, y_train, param_grid, cv_folds=config.CV_FOLDS):
    """
    Train a neural network classifier with hyperparameter optimization via grid search.

    Performs exhaustive search over specified parameter combinations using
    stratified k-fold cross-validation to find the best hyperparameters.

    Args:
        X_train: Training features array of shape (n_samples, n_features)
        y_train: Training labels array of shape (n_samples,)
        param_grid: Dictionary with parameter names as keys and lists of values to try
        cv_folds: Number of cross-validation folds

    Returns:
        tuple: (best_estimator, best_cv_score, best_params) containing the
               fitted model with optimal parameters and validation performance
    """
    print("\n--- NEURAL NET GRID SEARCH STARTING ---")
    print(f"Hyperparameters Grid: {param_grid}")
    print(f"Cross-Validation: {cv_folds} Folds")

    # Initialize stratified k-fold cross-validator
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True)
    model = NeuralNetwork()

    # Setup grid search with parallel processing
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=True
    )

    # Execute grid search
    grid.fit(X_train, y_train)

    print("\n--- NEURAL NET GRID SEARCH COMPLETED ---")
    print(f"Best Params: {grid.best_params_}")
    print(f"Best CV Score: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_score_, grid.best_params_