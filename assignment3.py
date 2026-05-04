# Name this file assignment3.py when you submit
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

# PyTorch dataset for the UWaveGestureLibrary dataset
class UWaveGestureLibraryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_filepath):
        X_list = []
        y_list = []

        with open(dataset_filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(":")
                if len(parts) < 4:
                    raise ValueError(f"Line {line_num}: expected 4 colon-separated groups, got {len(parts)}")

                x_str, y_str, z_str, label_str = parts[0], parts[1], parts[2], parts[3]

                # Convert comma-separated numbers efficiently
                x = np.fromstring(x_str, sep=",", dtype=np.float32)
                y = np.fromstring(y_str, sep=",", dtype=np.float32)
                z = np.fromstring(z_str, sep=",", dtype=np.float32)

                if x.size != 315 or y.size != 315 or z.size != 315:
                    raise ValueError(
                        f"Line {line_num}: expected 315 values each for x/y/z, got {x.size}/{y.size}/{z.size}"
                    )

                # Label might have commas/spaces; take the first clean token
                label_str = label_str.strip().split(",")[0]
                label = int(float(label_str))

                feats = np.concatenate([x, y, z])  # shape (945,)
                X_list.append(feats)
                y_list.append(label)

        self.features = torch.tensor(np.stack(X_list), dtype=torch.float32)
        self.labels = torch.tensor(np.array(y_list) - 1, dtype=torch.int64) # change labels to 0 to 7 

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
       x = self.features[idx].view(3, 315)
       # Convert label (0–7) to one-hot vector of size 8
       y_onehot = torch.nn.functional.one_hot(self.labels[idx], num_classes=8 ).float()

       return x, y_onehot


# A function that creates a cnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_cnn_model(training_data_filepath):
    # Load dataset using the provided filepath
    dataset = UWaveGestureLibraryDataset(training_data_filepath)

    # divide dataset into 75 percent train and 25 perecentn validation
    N = len(dataset)
    mid = int(0.75*N)
    dataset_train, dataset_validation = random_split(dataset, [mid, N - mid])

    # Create loaders 
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset_validation, batch_size=32, shuffle=False)

    # CNN class
    class UWaveCNN(nn.Module):
        def __init__(self, num_classes=8, ks=(21,15,9)):
            super().__init__()
            k1,k2,k3 = ks

            self.conv1 = nn.Conv1d(3, 32, kernel_size=k1, padding=k1//2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=k2, padding=k2//2)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=k3, padding=k3//2)

            self.pool = nn.MaxPool1d(kernel_size=2)

            self.adapt = nn.AdaptiveAvgPool1d(39)

            self.fc1 = nn.Linear(128 * 39, 256)
            self.fc2 = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # (B, 32, 157)
            x = self.pool(F.relu(self.conv2(x)))  # (B, 64, 78)
            x = self.pool(F.relu(self.conv3(x)))  # (B, 128, 39)

            x = self.adapt(x)

            x = torch.flatten(x, 1)               # (B, 128*39)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Instantiate model (OUTSIDE the class)
    model = UWaveCNN(num_classes=8)

    # Loss + optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # because we monitor validation accuracy
    factor=0.5,        # reduce LR by half
    patience=2,        # wait 2 epochs without improvement
    )  

    # ---- Training Loop ----
    for epoch in range(10):
        model.train()
        total_correct = 0
        total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            y_idx = torch.argmax(y, dim=1)      # one-hot -> index
            loss = loss_fn(outputs, y_idx)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(y, dim=1)
            total_correct += (preds == true_labels).sum().item()
            total += y.size(0)

        train_acc = total_correct / total

        # ---- Validation each epoch ----
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
          for x, y in val_loader:
             outputs = model(x)
             preds = torch.argmax(outputs, dim=1)
             true_labels = torch.argmax(y, dim=1)   # <-- convert one-hot to indices
             val_correct += (preds == true_labels).sum().item()
             val_total += y.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        scheduler.step(val_acc)

    training_performance = train_acc
    validation_performance = val_acc
    return model, training_performance, validation_performance

# A function that creates an rnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_rnn_model(training_data_filepath):
    dataset = UWaveGestureLibraryDataset(training_data_filepath)

    # 75/25 split
    N = len(dataset)
    mid = int(0.75 * N)
    dataset_train, dataset_validation = random_split(dataset, [mid, N - mid])

    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset_validation, batch_size=32, shuffle=False)

    class UWaveRNN(nn.Module):
        def __init__(self, input_size=3, hidden_size=256, num_layers=2, num_classes=8, dropout=0.2):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=True
            )
            self.fc1 = nn.Linear(hidden_size * 2, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            # (B,3,315) -> (B,315,3)
            x = x.transpose(1, 2)

            out, _ = self.gru(x)      # (B,315,2H)

            # MANY-TO-ONE via temporal pooling 
            feat = out.mean(dim=1)    # (B,2H)

            z = F.relu(self.fc1(feat))
            return self.fc2(z)

    model = UWaveRNN(num_classes=8)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # For such a tiny dataset, either skip scheduler or make it very patient
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    for epoch in range(30):
        model.train()
        total_correct, total = 0, 0

        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            y_idx = torch.argmax(y, dim=1)      # one-hot -> index
            loss = loss_fn(outputs, y_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # helps RNN stability
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(y, dim=1)
            total_correct += (preds == true_labels).sum().item()
            total += y.size(0)

        train_acc = total_correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(y, dim=1)   # <-- convert one-hot to indices
                val_correct += (preds == true_labels).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        scheduler.step(val_acc)

    training_performance = train_acc
    validation_performance = val_acc
    return model, training_performance, validation_performance



if __name__ == "__main__":

    def evaluate_model(model, dataloader):
        model.eval()
        total_correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(y, dim=1)
                total_correct += (preds == true_labels).sum().item()
                total += y.size(0)
        return total_correct / total

    model1, tp1, vp1 = u_wave_gesture_library_cnn_model("UWaveGestureLibrary_TRAIN.csv")
    model2, tp2, vp2 = u_wave_gesture_library_rnn_model("UWaveGestureLibrary_TRAIN.csv")

    test_dataset = UWaveGestureLibraryDataset("UWaveGestureLibrary_TEST.csv")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    cnn_test_acc = evaluate_model(model1, test_loader)
    rnn_test_acc = evaluate_model(model2, test_loader)

    print("CNN Test Accuracy:", cnn_test_acc)
    print("RNN Test Accuracy:", rnn_test_acc)