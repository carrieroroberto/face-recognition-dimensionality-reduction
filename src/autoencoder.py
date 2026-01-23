# src/autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class FaceAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FaceAutoencoder, self).__init__()

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, latent_dim)
        )

        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # output tra 0 e 1
        )

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

    # --- Fit ---
    def fit(self, X_train, epochs=200, batch_size=32, lr=1e-3):
        """
        Addestra l'autoencoder.
        Restituisce self e loss_history.
        """
        # Normalizzazione
        X_train = X_train / 255.0 if X_train.max() > 1 else X_train

        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        loss_history = []

        print(f"Inizio training Autoencoder ({epochs} epoche)...")
        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for batch_X, _ in loader:
                optimizer.zero_grad()
                reconstructed, _ = self(batch_X)
                loss = criterion(reconstructed, batch_X)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)

            avg_loss = train_loss / len(loader.dataset)
            loss_history.append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

        return self, loss_history

    # --- Trasformazione per ottenere features latenti ---
    def transform(self, X):
        """
        Restituisce la rappresentazione latente (features estratte).
        """
        self.eval()
        X = X / 255.0 if X.max() > 1 else X
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            _, latent = self(X_tensor)
        return latent.numpy()

    # --- Ricostruzione immagini ---
    def reconstruct(self, X):
        """
        Restituisce la ricostruzione dell'input X.
        """
        self.eval()
        X = X / 255.0 if X.max() > 1 else X
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            reconstructed, _ = self(X_tensor)
        return reconstructed.numpy()