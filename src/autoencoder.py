# src/autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import config

class ImprovedFaceAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers=None,
                 use_batch_norm=True, dropout_rate=0.2):
        super(ImprovedFaceAutoencoder, self).__init__()

        if hidden_layers is None:
            hidden_layers = config.AE_HIDDEN_LAYERS

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def fit(self, X_train, X_val=None, epochs=None, batch_size=None,
            lr=None, weight_decay=None, patience=None, verbose=True):
        if epochs is None:
            epochs = config.AE_EPOCHS
        if batch_size is None:
            batch_size = config.AE_BATCH_SIZE
        if lr is None:
            lr = config.AE_LEARNING_RATE
        if weight_decay is None:
            weight_decay = config.AE_WEIGHT_DECAY
        if patience is None:
            patience = config.AE_PATIENCE

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(X_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(X_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        history = {
            'train_loss': [],
            'val_loss': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        if verbose:
            print(f"\nTraining Autoencoder:")
            print(f"  Epochs: {epochs}, Batch Size: {batch_size}")
            print(f"  Learning Rate: {lr}, Weight Decay: {weight_decay}")
            print(f"  Architecture: {self.input_dim} -> {config.AE_HIDDEN_LAYERS} -> {self.latent_dim}")
            print(f"  Batch Norm: {self.use_batch_norm}, Dropout: {self.dropout_rate}")

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for batch_X, _ in train_loader:
                optimizer.zero_grad()
                reconstructed, _ = self(batch_X)
                loss = criterion(reconstructed, batch_X)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)
            history['train_loss'].append(avg_train_loss)

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, _ in val_loader:
                        reconstructed, _ = self(batch_X)
                        loss = criterion(reconstructed, batch_X)
                        val_loss += loss.item() * batch_X.size(0)

                avg_val_loss = val_loss / len(val_loader.dataset)
                history['val_loss'].append(avg_val_loss)

                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_state = self.state_dict().copy()
                else:
                    patience_counter += 1

                if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {avg_val_loss:.6f}")

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    self.load_state_dict(best_state)
                    break
            else:
                if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}")

        if verbose:
            print("Training completato!\n")

        return self, history

    def transform(self, X):
        self.eval()
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            _, latent = self(X_tensor)
        return latent.numpy()

    def reconstruct(self, X):
        self.eval()
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            reconstructed, _ = self(X_tensor)
        return reconstructed.numpy()

    def compute_reconstruction_error(self, X):
        X_reconstructed = self.reconstruct(X)
        mse = np.mean((X - X_reconstructed) ** 2)
        return mse