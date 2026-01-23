# src/classification.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
import numpy as np
import config


def train_svm_with_grid_search(X_train, y_train, param_grid=None, cv_folds=None, verbose=True):
    if param_grid is None:
        param_grid = config.SVM_PARAM_GRID
    if cv_folds is None:
        cv_folds = config.SVM_CV_FOLDS

    if verbose:
        print("SVM Grid Search")
        print(f"  Parametri: {param_grid}")
        print(f"  CV: {cv_folds} folds")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

    grid = GridSearchCV(
        SVC(probability=True, class_weight='balanced', random_state=config.RANDOM_STATE),
        param_grid, cv=cv, scoring='accuracy',
        n_jobs=config.CV_N_JOBS, verbose=0, return_train_score=True
    )
    grid.fit(X_train, y_train)

    if verbose:
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV score: {grid.best_score_:.4f}")

    return grid.best_estimator_, grid.best_score_, grid.best_params_


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_layers=None, dropout=None):
        if hidden_layers is None:
            hidden_layers = config.NN_HIDDEN_LAYERS
        if dropout is None:
            dropout = config.NN_DROPOUT
        super(NeuralNetwork, self).__init__()

        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            epochs=None, batch_size=None, lr=None, weight_decay=None,
            patience=None, use_class_weights=None, verbose=True):
        if epochs is None:
            epochs = config.NN_EPOCHS
        if batch_size is None:
            batch_size = config.NN_BATCH_SIZE
        if patience is None:
            patience = config.NN_PATIENCE
        if use_class_weights is None:
            use_class_weights = config.NN_USE_CLASS_WEIGHTS

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Class weights per bilanciare classi sbilanciate
        if use_class_weights:
            class_counts = np.bincount(y_train)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_weights)
            criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.NN_LR_FACTOR, patience=config.NN_LR_PATIENCE
        )

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        if verbose:
            print(f"  Training NN: {self.input_dim} -> {self.hidden_layers} -> {self.n_classes}")
            print(f"    LR: {lr}, WD: {weight_decay}")

        for epoch in range(epochs):
            self.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

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
                val_loss, val_correct, val_total = 0.0, 0, 0

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

                if patience_counter >= patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch+1}")
                    self.load_state_dict(best_state)
                    break

        return self, history

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(torch.FloatTensor(X))
            _, preds = torch.max(outputs, 1)
        return preds.numpy()

    def predict_proba(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self(torch.FloatTensor(X))
            probs = torch.softmax(outputs, dim=1)
        return probs.numpy()


def train_nn_with_grid_search(X_train, y_train, n_classes, X_val=None, y_val=None,
                               param_grid=None, verbose=True):
    if param_grid is None:
        param_grid = config.NN_PARAM_GRID

    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=config.RANDOM_STATE
        )

    if verbose:
        print("Neural Network Grid Search")
        print(f"  Parametri: {param_grid}")

    best_score = -np.inf
    best_params = None
    best_model = None

    param_combinations = list(product(
        param_grid['learning_rate'],
        param_grid['weight_decay']
    ))

    total = len(param_combinations)

    for idx, (lr, wd) in enumerate(param_combinations, 1):
        if verbose:
            print(f"  [{idx}/{total}] LR={lr}, WD={wd}")

        model = NeuralNetwork(
            input_dim=X_train.shape[1],
            n_classes=n_classes
        )

        model, _ = model.fit(
            X_train, y_train, X_val=X_val, y_val=y_val,
            lr=lr, weight_decay=wd, verbose=False
        )

        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)

        if verbose:
            print(f"    Val Accuracy: {val_acc:.4f}")

        if val_acc > best_score:
            best_score = val_acc
            best_params = {'learning_rate': lr, 'weight_decay': wd}
            best_model = model

    if verbose:
        print(f"  Best params: {best_params}")
        print(f"  Best val accuracy: {best_score:.4f}")

    return best_model, best_params, best_score
