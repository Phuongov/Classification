import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = "./data/raw"
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 25
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
NUM_WORKERS = 4
MODEL_SAVE_PATH = "./results/resnet34_best.pth"

# ============================================================
# Set Seed
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)

# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# Data Transformations
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# Dataset
# ============================================================

full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transforms
)

classes = full_dataset.classes
print(f"Classes: {classes}")

# ============================================================
# Train / Validation / Test Split
# ============================================================

train_size = int(0.7 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size]
)

# Validation/Test should not use augmentation
val_dataset.dataset.transform = val_transforms
test_dataset.dataset.transform = val_transforms

print(f"Train size: {len(train_dataset)}")
print(f"Validation size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ============================================================
# ResNet34 Model
# ============================================================

model = models.resnet34(pretrained=True)

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))

model = model.to(device)

# ============================================================
# Loss and Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ============================================================
# Training Function
# ============================================================


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, leave=True)

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loop.set_description("Training")
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1

# ============================================================
# Validation Function
# ============================================================


def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels

# ============================================================
# Training Loop
# ============================================================

best_val_acc = 0
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc, train_f1 = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_acc, val_f1, _, _ = evaluate(
        model,
        val_loader,
        criterion
    )

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Train Macro F1: {train_f1:.4f}")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Macro F1: {val_f1:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc

        os.makedirs("./results", exist_ok=True)

        torch.save(model.state_dict(), MODEL_SAVE_PATH)

        print("Best model saved.")

# ============================================================
# Final Test Evaluation
# ============================================================

print("\nLoading best model...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

print("Evaluating on test set...")

_, test_acc, test_f1, test_preds, test_labels = evaluate(
    model,
    test_loader,
    criterion
)

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Macro F1: {test_f1:.4f}")

# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report")
print(
    classification_report(
        test_labels,
        test_preds,
        target_names=classes
    )
)

# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(test_labels, test_preds)

print("\nConfusion Matrix")
print(cm)

print("\nTraining completed successfully.")
