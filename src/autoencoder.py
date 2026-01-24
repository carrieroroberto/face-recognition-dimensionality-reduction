"""
This module implements a fully-connected autoencoder neural network for
learning compressed representations of face images. The autoencoder consists
of an encoder that maps high-dimensional input to a low-dimensional latent
space, and a decoder that reconstructs the original input from the latent
representation.

Key features:
- Configurable architecture with variable hidden layers and latent dimensions
- Batch normalization and dropout for regularization
- Early stopping to prevent overfitting
- Support for feature extraction and image reconstruction
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
import config
from src.utils import plot_training_loss, plot_ablation_study


class Autoencoder(nn.Module):
    """
    Fully-connected autoencoder for dimensionality reduction.

    The network architecture consists of:
    - Encoder: Input -> Hidden layers -> Latent space
    - Decoder: Latent space -> Hidden layers (reversed) -> Reconstructed output

    Attributes:
        input_dim: Dimensionality of input features
        latent_dim: Size of the latent representation (bottleneck)
        encoder: Sequential module for encoding
        decoder: Sequential module for decoding
    """

    def __init__(self, input_dim, latent_dim, hidden_layers=config.AE_HIDDEN_LAYERS, input_dropout=config.AE_DROPOUT_RATE, verbose=True):
        """
        Initialize the autoencoder architecture.

        Args:
            input_dim: Number of input features (flattened image pixels)
            latent_dim: Dimension of the latent space representation
            hidden_layers: List of hidden layer sizes for encoder
            input_dropout: Dropout rate applied to input layer
            verbose: Whether to print training progress
        """
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.verbose = verbose

        # Build encoder: progressively reduce dimensionality
        encoder_layers = []

        # Apply dropout to input for regularization
        if input_dropout > 0:
            encoder_layers.append(nn.Dropout(p=input_dropout))

        # Add hidden layers with batch normalization and LeakyReLU activation
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim

        # Final projection to latent space
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder: mirror the encoder architecture
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.Dropout(input_dropout))
            prev_dim = hidden_dim

        # Final reconstruction layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights using Xavier uniform initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.

        This initialization helps maintain gradient flow during training
        by keeping the variance of activations roughly constant across layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            tuple: (reconstructed, latent) where reconstructed is the
                   decoded output and latent is the bottleneck representation
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def fit(self, X_train, X_val=None, epochs=config.AE_EPOCHS,
            batch_size=config.AE_BATCH_SIZE, lr=1e-3, weight_decay=1e-4, patience=config.AE_PATIENCE):
        """
        Train the autoencoder on the given data.

        Uses MSE loss to minimize reconstruction error and Adam optimizer
        for weight updates. Implements early stopping based on validation loss.

        Args:
            X_train: Training data array of shape (n_samples, n_features)
            X_val: Optional validation data for early stopping
            epochs: Maximum number of training epochs
            batch_size: Mini-batch size for training
            lr: Learning rate for Adam optimizer
            weight_decay: L2 regularization coefficient
            patience: Number of epochs without improvement before early stopping

        Returns:
            tuple: (self, history) where history contains training/validation losses
        """
        # Create PyTorch DataLoader for mini-batch training
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Create validation DataLoader if validation data provided
        val_loader = None
        if X_val is not None:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Initialize training history tracking
        history = {"train_loss": [], "val_loss": []}

        # Early stopping state
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_weights = copy.deepcopy(self.state_dict())

        if self.verbose:
            print(f"Training AE (Latent: {self.latent_dim})...")

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            for batch_X, target_X in train_loader:
                optimizer.zero_grad()
                reconstructed, _ = self(batch_X)
                loss = criterion(reconstructed, target_X)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)

            # Compute average training loss
            avg_train_loss = train_loss / len(train_loader.dataset)
            history["train_loss"].append(avg_train_loss)

            # Validation phase
            avg_val_loss = 0.0
            if val_loader:
                self.eval()
                val_loss_accum = 0.0
                with torch.no_grad():
                    for batch_X, target_X in val_loader:
                        reconstructed, _ = self(batch_X)
                        loss = criterion(reconstructed, target_X)
                        val_loss_accum += loss.item() * batch_X.size(0)

                avg_val_loss = val_loss_accum / len(val_loader.dataset)
                history["val_loss"].append(avg_val_loss)

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_weights = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if self.verbose:
                            print(f"  Early stopping at epoch {epoch+1}")
                        self.load_state_dict(best_model_weights)
                        break

            # Print progress at regular intervals
            if self.verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                val_str = f" | Val: {avg_val_loss:.5f}" if val_loader else ""
                print(f"  Epoch: {epoch+1}/{epochs} | Train: {avg_train_loss:.5f}{val_str}")

        return self, history

    def transform(self, X):
        """
        Encode input data to latent space representation.

        Args:
            X: Input array of shape (n_samples, n_features)

        Returns:
            numpy.ndarray: Latent representations of shape (n_samples, latent_dim)
        """
        self.eval()
        with torch.no_grad():
            _, latent = self(torch.FloatTensor(X))
        return latent.numpy()

    def reconstruct(self, X):
        """
        Reconstruct input data through the full autoencoder.

        Args:
            X: Input array of shape (n_samples, n_features)

        Returns:
            numpy.ndarray: Reconstructed data of shape (n_samples, n_features)
        """
        self.eval()
        with torch.no_grad():
            reconstructed, _ = self(torch.FloatTensor(X))
        return reconstructed.numpy()

    def compute_reconstruction_error(self, X):
        """
        Compute mean squared reconstruction error on given data.

        Args:
            X: Input array of shape (n_samples, n_features)

        Returns:
            float: Mean squared error between input and reconstruction
        """
        self.eval()
        with torch.no_grad():
            X_torch = torch.FloatTensor(X)
            rec, _ = self(X_torch)
            mse = nn.MSELoss()(rec, X_torch).item()
        return mse


def run_autoencoder_experiments(latent_dims_list, X_train, X_test, y_train, y_test, h, w, verbose=True):
    """
    Run ablation study over different latent space dimensions.

    Trains autoencoders with varying latent dimensions and evaluates
    their performance using SVM classification on the learned features.

    Args:
        latent_dims_list: List of latent dimension values to test
        X_train: Training features array
        X_test: Test features array
        y_train: Training labels
        y_test: Test labels
        h: Image height (for visualization)
        w: Image width (for visualization)
        verbose: Whether to print progress

    Returns:
        dict: Results for each latent dimension containing model, features,
              reconstruction error, and classification accuracy
    """
    results = {}
    test_accuracies = []
    test_mses = []

    # Iterate over each latent dimension configuration
    for latent_dim in latent_dims_list:
        if verbose:
            print(f"\n--- AE Latent Dim: {latent_dim} ---")

        # Initialize autoencoder with current latent dimension
        final_ae = Autoencoder(
            input_dim=X_train.shape[1],
            latent_dim=latent_dim,
            hidden_layers=config.AE_HIDDEN_LAYERS,
            input_dropout=config.AE_DROPOUT_RATE,
            verbose=verbose
        )

        # Train the autoencoder
        final_ae, history = final_ae.fit(
            X_train, X_val=X_test,
            epochs=config.AE_EPOCHS,
            batch_size=config.AE_BATCH_SIZE,
            lr=config.AE_LEARNING_RATE,
            weight_decay=config.AE_WEIGHT_DECAY,
            patience=config.AE_PATIENCE
        )

        # Extract latent features for classification
        X_train_feat = final_ae.transform(X_train)
        X_test_feat = final_ae.transform(X_test)

        # Compute reconstruction quality
        final_mse = final_ae.compute_reconstruction_error(X_test)

        # Evaluate feature quality using SVM classifier
        svm_final = SVC(kernel="rbf")
        svm_final.fit(X_train_feat, y_train)
        final_acc = svm_final.score(X_test_feat, y_test)

        test_accuracies.append(final_acc)
        test_mses.append(final_mse)

        # Plot training loss curve
        if verbose:
            plot_training_loss(
                history["train_loss"],
                history["val_loss"],
                title=f"AE Training (Dim {latent_dim})"
            )

        # Store results for this configuration
        results[latent_dim] = {
            "model": final_ae,
            "history": history,
            "X_train_feat": X_train_feat,
            "X_test_feat": X_test_feat,
            "test_mse": final_mse,
            "test_acc": final_acc
        }

    # Generate ablation study visualization
    plot_ablation_study(
        x_values=latent_dims_list,
        accuracy_scores=test_accuracies,
        secondary_metric_scores=test_mses,
        x_label="Latent Dimension Size",
        secondary_label="Reconstruction MSE",
        title="Ablation Study: Autoencoder Latent Dim vs Performance"
    )

    return results