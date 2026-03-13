import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import random_split, DataLoader

from src.datasets.ntu_dataset import NTUDataset
from src.classifiers.ntu_baseline import NTUBaselineClassifier


DATA_DIR = "data/ntu_subset"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "ntu_baseline.pt")

BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-3
VAL_RATIO = 0.2
RANDOM_SEED = 42

NUM_CLASSES = 5
NUM_JOINTS = 17


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_samples, 1)

    return avg_loss, accuracy


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_samples, 1)

    return avg_loss, accuracy


def main():
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = NTUDataset(DATA_DIR)
    dataset_size = len(dataset)

    if dataset_size == 0:
        raise ValueError(f"No samples found in {DATA_DIR}")

    val_size = max(1, int(dataset_size * VAL_RATIO))
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = NTUBaselineClassifier(
        num_joints=NUM_JOINTS,
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(MODEL_DIR, exist_ok=True)

    best_val_acc = 0.0

    print(f"Dataset size: {dataset_size}")
    print(f"Train size:   {train_size}")
    print(f"Val size:     {val_size}")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH}")

    print(f"Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
