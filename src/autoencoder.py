import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
import config
from src.utils import plot_training_loss, plot_ablation_study

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layers=config.AE_HIDDEN_LAYERS, input_dropout=config.AE_DROPOUT_RATE, verbose=True):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim 
        self.latent_dim = latent_dim
        self.verbose = verbose
        
        encoder_layers = []
        
        if input_dropout > 0:
            encoder_layers.append(nn.Dropout(p=input_dropout))
            
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim
            
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.Dropout(input_dropout))
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def fit(self, X_train, X_val=None, epochs=config.AE_EPOCHS, 
            batch_size=config.AE_BATCH_SIZE, lr=1e-3, weight_decay=1e-4, patience=config.AE_PATIENCE):
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        history = {"train_loss": [], "val_loss": []}
        
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_weights = copy.deepcopy(self.state_dict())

        if self.verbose:
            print(f"Training AE (Latent: {self.latent_dim})...")

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for batch_X, target_X in train_loader:
                optimizer.zero_grad()
                reconstructed, _ = self(batch_X)
                loss = criterion(reconstructed, target_X)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            history["train_loss"].append(avg_train_loss)

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
            
            if self.verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                val_str = f" | Val: {avg_val_loss:.5f}" if val_loader else ""
                print(f"  Epoch: {epoch+1}/{epochs} | Train: {avg_train_loss:.5f}{val_str}")

        return self, history

    def transform(self, X):
        self.eval()
        with torch.no_grad():
            _, latent = self(torch.FloatTensor(X))
        return latent.numpy()

    def reconstruct(self, X):
        self.eval()
        with torch.no_grad():
            reconstructed, _ = self(torch.FloatTensor(X))
        return reconstructed.numpy()

    def compute_reconstruction_error(self, X):
        self.eval()
        with torch.no_grad():
            X_torch = torch.FloatTensor(X)
            rec, _ = self(X_torch)
            mse = nn.MSELoss()(rec, X_torch).item()
        return mse

def run_autoencoder_experiments(latent_dims_list, X_train, X_test, y_train, y_test, h, w, verbose=True):
    results = {}
    test_accuracies = []
    test_mses = []
    
    for latent_dim in latent_dims_list:
        if verbose:
            print(f"\n--- AE Latent Dim: {latent_dim} ---")
            
        final_ae = Autoencoder(
            input_dim=X_train.shape[1],
            latent_dim=latent_dim,
            hidden_layers=config.AE_HIDDEN_LAYERS,
            input_dropout=config.AE_DROPOUT_RATE,
            verbose=verbose
        )
        
        final_ae, history = final_ae.fit(
            X_train, X_val=X_test, 
            epochs=config.AE_EPOCHS, 
            batch_size=config.AE_BATCH_SIZE,
            lr=config.AE_LEARNING_RATE,
            weight_decay=config.AE_WEIGHT_DECAY,
            patience=config.AE_PATIENCE
        )

        X_train_feat = final_ae.transform(X_train)
        X_test_feat = final_ae.transform(X_test)
        
        final_mse = final_ae.compute_reconstruction_error(X_test)
        
        svm_final = SVC(kernel="rbf")
        svm_final.fit(X_train_feat, y_train)
        final_acc = svm_final.score(X_test_feat, y_test)
        
        test_accuracies.append(final_acc)
        test_mses.append(final_mse)

        if verbose:
            plot_training_loss(
                history["train_loss"], 
                history["val_loss"], 
                title=f"AE Training (Dim {latent_dim})"
            )

        results[latent_dim] = {
            "model": final_ae,
            "history": history,
            "X_train_feat": X_train_feat,
            "X_test_feat": X_test_feat,
            "test_mse": final_mse,
            "test_acc": final_acc
        }

    plot_ablation_study(
        x_values=latent_dims_list,
        accuracy_scores=test_accuracies,
        secondary_metric_scores=test_mses,
        x_label="Latent Dimension Size",
        secondary_label="Reconstruction MSE",
        title="Ablation Study: Autoencoder Latent Dim vs Performance"
    )

    return results