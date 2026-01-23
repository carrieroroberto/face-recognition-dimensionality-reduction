# src/classification.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import config

def train_svm_with_grid_search(X_train, y_train, param_grid=None, cv_folds=None, verbose=True):
    if param_grid is None:
        param_grid = config.SVM_PARAM_GRID
    if cv_folds is None:
        cv_folds = config.SVM_CV_FOLDS

    if verbose:
        print("GRID SEARCH SVM")
        print(f"Parametri da testare: {param_grid}")
        print(f"Cross-Validation: {cv_folds} folds")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

    grid = GridSearchCV(
        SVC(probability=True, random_state=config.RANDOM_STATE),
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=config.CV_N_JOBS,
        verbose=1 if verbose else 0,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    if verbose:
        print(f"\nMigliori parametri: {grid.best_params_}")
        print(f"Best CV Score: {grid.best_score_:.4f}")
        print(f"CV Std Dev: {grid.cv_results_['std_test_score'][grid.best_index_]:.4f}")
        print("="*60 + "\n")

    return grid.best_estimator_, grid.best_score_, grid.best_params_

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_layers=(128, 64), dropout_rate=0.3):
        super(ImprovedNeuralNetwork, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=None, batch_size=None, lr=None, weight_decay=None,
            patience=None, verbose=True):
        if epochs is None:
            epochs = config.NN_EPOCHS
        if batch_size is None:
            batch_size = config.NN_BATCH_SIZE
        if lr is None:
            lr = list(config.NN_PARAM_GRID['learning_rate'])[0]
        if weight_decay is None:
            weight_decay = list(config.NN_PARAM_GRID['weight_decay'])[0]
        if patience is None:
            patience = config.NN_PATIENCE

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.NN_LR_FACTOR,
            patience=config.NN_LR_PATIENCE
        )

        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        if verbose:
            print(f"\nTraining Neural Network:")
            print(f"  Architecture: {self.input_dim} -> {self.hidden_layers} -> {self.n_classes}")
            print(f"  Dropout: {self.dropout_rate}")
            print(f"  Epochs: {epochs}, Batch Size: {batch_size}")
            print(f"  Learning Rate: {lr}, Weight Decay: {weight_decay}")

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += y_batch.size(0)
                train_correct += (predicted == y_batch).sum().item()

            avg_train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / train_total
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = self(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += y_batch.size(0)
                        val_correct += (predicted == y_batch).sum().item()

                avg_val_loss = val_loss / len(val_loader.dataset)
                val_acc = val_correct / val_total
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_acc)

                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_state = self.state_dict().copy()
                else:
                    patience_counter += 1

                if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    self.load_state_dict(best_state)
                    break
            else:
                if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                    print(f"Epoch [{epoch+1}/{epochs}] - "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

        if verbose:
            print("Training completato!\n")

        return self, history

    def predict(self, X):
        """Predizioni classe."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self(X_tensor)
            _, preds = torch.max(outputs, 1)
        return preds.numpy()

    def predict_proba(self, X):
        """Probabilità predette."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.numpy()

def train_nn_with_grid_search(X_train, y_train, n_classes, X_val=None, y_val=None,
                               param_grid=None, verbose=True):
    if param_grid is None:
        param_grid = config.NN_PARAM_GRID

    if X_val is None or y_val is None:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train,
            random_state=config.RANDOM_STATE
        )

    if verbose:
        print("\n" + "="*60)
        print("GRID SEARCH NEURAL NETWORK")
        print("="*60)
        print(f"Parametri da testare: {param_grid}")

    best_score = -np.inf
    best_params = None
    best_model = None

    from itertools import product
    param_combinations = list(product(
        param_grid['hidden_layers'],
        param_grid['learning_rate'],
        param_grid['dropout_rate'],
        param_grid['weight_decay']
    ))

    total_combinations = len(param_combinations)

    if verbose:
        print(f"Totale combinazioni da testare: {total_combinations}\n")

    for idx, (hidden_layers, lr, dropout, wd) in enumerate(param_combinations, 1):
        if verbose:
            print(f"[{idx}/{total_combinations}] Testing: hidden={hidden_layers}, lr={lr}, "
                  f"dropout={dropout}, wd={wd}")

        model = ImprovedNeuralNetwork(
            input_dim=X_train.shape[1],
            n_classes=n_classes,
            hidden_layers=hidden_layers,
            dropout_rate=dropout
        )

        model, history = model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            lr=lr,
            weight_decay=wd,
            verbose=False
        )

        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        if verbose:
            print(f"  Validation Accuracy: {val_acc:.4f}\n")

        if val_acc > best_score:
            best_score = val_acc
            best_params = {
                'hidden_layers': hidden_layers,
                'learning_rate': lr,
                'dropout_rate': dropout,
                'weight_decay': wd
            }
            best_model = model

    if verbose:
        print("="*60)
        print(f"Migliori parametri: {best_params}")
        print(f"Best Validation Accuracy: {best_score:.4f}")
        print("="*60 + "\n")

    return best_model, best_params, best_score