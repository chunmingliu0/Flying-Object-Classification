import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ===== Load preprocessed train/val data for different SNR levels =====
X_train4 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/X_train_Fz2_SNR10.npy')
y_train4 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/y_train_Fz2_SNR10.npy')
X_val4 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/X_val_Fz2_SNR10.npy')
y_val4 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/y_val_Fz2_SNR10.npy')

X_train3 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/X_train_Fz2_SNR20.npy')
y_train3 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/y_train_Fz2_SNR20.npy')
X_val3 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/X_val_Fz2_SNR20.npy')
y_val3 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/y_val_Fz2_SNR20.npy')

X_train2 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/X_train_Fz2_SNR30.npy')
y_train2 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/y_train_Fz2_SNR30.npy')
X_val2 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/X_val_Fz2_SNR30.npy')
y_val2 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/y_val_Fz2_SNR30.npy')

X_train1 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/X_train_Fz2_SNR40.npy')
y_train1 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/y_train_Fz2_SNR40.npy')
X_val1 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/X_val_Fz2_SNR40.npy')
y_val1 = np.load('/content/drive/MyDrive/MPE_Research/Data/Train/y_val_Fz2_SNR40.npy')

# ===== Merge all SNR-level datasets =====
X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4), axis=0)
y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4), axis=0)
X_val = np.concatenate((X_val1, X_val2, X_val3, X_val4), axis=0)
y_val = np.concatenate((y_val1, y_val2, y_val3, y_val4), axis=0)

# Check shapes
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print(f"Sample original labels: {y_train[:5]}")

from sklearn.preprocessing import LabelEncoder

# Encode string labels (e.g., 'class1', 'class2') into integers
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

from scipy.spatial.distance import jensenshannon

# ===== Calculate Jensen-Shannon distances between class-mean feature distributions =====
num_classes = len(np.unique(y_train))
js_matrix = np.zeros((num_classes, num_classes))

# Group samples by class
class_samples = [X_train[y_train == i] for i in range(num_classes)]

# Compute mean vector per class and normalize into probability distribution
mean_distributions = []
for i in range(num_classes):
    mean_vec = np.mean(class_samples[i], axis=0)
    mean_vec = np.abs(mean_vec)
    mean_vec /= np.sum(mean_vec) + 1e-10
    mean_distributions.append(mean_vec)

# Calculate pairwise Jensen-Shannon distances
for i in range(num_classes):
    for j in range(i + 1, num_classes):
        js = jensenshannon(mean_distributions[i], mean_distributions[j])
        js_matrix[i][j] = js
        js_matrix[j][i] = js  # Symmetric matrix

# Visualize JS distances as heatmap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_js = pd.DataFrame(js_matrix, columns=[f'Class {i}' for i in range(num_classes)],
                               index=[f'Class {i}' for i in range(num_classes)])
sns.heatmap(df_js, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Jensen-Shannon Distance Between Classes (X_train)")
plt.show()

# ===== Calculate intra-class variance =====
intra_class_variances = {}
for cls in range(num_classes):
    samples = X_train[y_train == cls]
    if len(samples) == 0:
        print(f"Warning: Class {cls} is empty.")
        intra_class_variances[f"Class {cls}"] = np.nan
        continue
    variance = np.var(samples, axis=0)
    mean_variance = np.mean(variance)
    intra_class_variances[f"Class {cls}"] = mean_variance

for cls, var in intra_class_variances.items():
    print(f"{cls} - Mean Intra-Class Variance: {var:.6f}")

# ===== Convert to PyTorch tensors =====
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train_encoded = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val_encoded, dtype=torch.long)
print(f"Encoded label samples: {y_train_encoded[:5]}")

# ===== Custom PyTorch Dataset =====
class SignalDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.unsqueeze(1)  # Add channel dimension: (batch, 1, signal_length)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
train_dataset = SignalDataset(X_train, y_train_encoded)
val_dataset = SignalDataset(X_val, y_val)
print("X_train shape:", train_dataset.X.shape)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# ===== CNN + BiLSTM feature extraction module =====
class EnhancedCNNBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # → [B, 64, 588]

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # → [B, 128, 294]

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),  # → [B, 256, 147]
        )

        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            dropout=0.4,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        x = self.cnn(x)             # → [B, 256, 147]
        x = x.permute(0, 2, 1)      # → [B, 147, 256]
        x, _ = self.bilstm(x)       # → [B, 147, 256]
        return x

# ===== Attention + perturbation-aware weighting + residual connection =====
class AttentionCNNBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = EnhancedCNNBiLSTM()

        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Per-timestep attention (no global normalization)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)  # → [B, 147, 256]

        # Base attention weights
        raw_attn = self.attention(x)  # → [B, 147, 1]

        # Perturbation-aware weighting (second-order difference)
        grad = torch.abs(x[:, 2:, :] - 2 * x[:, 1:-1, :] + x[:, :-2, :])
        perturb = grad.mean(dim=2, keepdim=True)  # → [B, 145, 1]
        perturb = F.interpolate(perturb.permute(0, 2, 1), size=147, mode='linear', align_corners=True)
        perturb = perturb.permute(0, 2, 1)  # → [B, 147, 1]

        # Combine base attention with perturbation guide
        attn_weights = torch.sigmoid(raw_attn + perturb)
        attn_context = torch.sum(attn_weights * x, dim=1)  # → [B, 256]

        # Residual connection using last timestep
        residual = x[:, -1, :]
        context = attn_context + residual

        return self.classifier(context)

# ===== Initialize model =====
num_classes = len(torch.unique(y_train_encoded))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model1 = AttentionCNNBiLSTM(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model1.parameters(), lr=0.0001)

# ===== Early stopping with dual metrics (accuracy + loss) =====
class DualMetricEarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_val_acc = None
        self.best_val_loss = None
        self.early_stop = False

    def __call__(self, val_acc, val_loss):
        if self.best_val_acc is None or self.best_val_loss is None:
            self.best_val_acc = val_acc
            self.best_val_loss = val_loss
            return

        acc_no_improve = val_acc < self.best_val_acc + self.delta
        loss_worsening = val_loss > self.best_val_loss - self.delta

        if acc_no_improve and loss_worsening:
            self.counter += 1
            print(f"⚠️ EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss

# ===== Training function =====
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, model_name="model", patience=5):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    best_model_path = f"/content/drive/MyDrive/MPE_Research/best_{model_name}.pth"

    early_stopper = DualMetricEarlyStopping(patience=patience, delta=1e-4)

    for epoch in range(num_epochs):
        train_loss, val_loss = 0, 0
        correct_train, total_train = 0, 0
        correct_val, total_val = 0, 0

        # ---- Training phase ----
        model.train()
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / total_train
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)

        # ---- Validation phase ----
        model.eval()
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ {model_name} improved! Saving best model with Val Acc: {val_acc:.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

        early_stopper(val_acc, val_loss / len(val_loader))
        if early_stopper.early_stop:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}. Best Val Acc: {best_val_acc:.4f}")
            break

    # ---- Plot curves ----
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{model_name} Training & Validation Loss Curve")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"{model_name} Training & Validation Accuracy Curve")
    plt.show()

# Train model
train_model(model1, train_loader, val_loader, criterion, optimizer, num_epochs=100, model_name="CNN_BiLSTM_Att_Fz2", patience=5)

# ===== Testing function =====
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def test_model(model, test_loader, model_name="Model"):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)

            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    print(f"✅ Final Test Accuracy ({model_name}): {accuracy:.4f}")

    print(f"\n📌 Classification Report ({model_name}):\n")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

# Load best model
model1.load_state_dict(torch.load("/content/drive/MyDrive/MPE_Research/best_CNN_BiLSTM_Att_Fz2.pth"))
model1.to(device)
print("✅ Model loaded successfully.")

# Load test set (SNR=40 dB)
X_test = np.load('/content/drive/MyDrive/MPE_Research/Data/Test/X_test_Fz2_SNR40.npy')
y_test = np.load('/content/drive/MyDrive/MPE_Research/Data/Test/y_test_Fz2_SNR40.npy')

y_test_encoded = label_encoder.transform(y_test)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test_encoded, dtype=torch.long)
test_dataset = SignalDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Test model
test_model(model1, test_loader, model_name="CNN_BiLSTM_Att under SNR 40 dB")
