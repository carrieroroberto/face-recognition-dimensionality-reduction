import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import config

# -------------------
# SVM
# -------------------
def train_svm(X_train, y_train):
    """Addestra una SVM con ricerca degli iperparametri (Grid Search)."""
    print("Ottimizzazione SVM via Grid Search...")

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"Migliori parametri SVM: {grid.best_params_}")
    return grid.best_estimator_

# -------------------
# Simple Neural Network con PyTorch
# -------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(NeuralNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.model(x)

    def fit(self, X_train, y_train, epochs=200, batch_size=32, lr=1e-3, wd=1e-5):
        """Addestra la rete neurale."""
        print("Inizio training NN (CPU)...")
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * X_batch.size(0)

            epoch_loss = running_loss / len(loader.dataset)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}")

        return self

    def predict(self, X):
        """Restituisce le classi predette."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self(X_tensor)
            preds = torch.argmax(outputs, dim=1).numpy()
        return preds

    def predict_proba(self, X):
        """Restituisce le probabilità predette."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self(X_tensor)
            probs = torch.softmax(outputs, dim=1).numpy()
        return probs