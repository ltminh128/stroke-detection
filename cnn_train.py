"""
Stroke Detection - CNN Fine-tuning with ResNet-50
==================================================
Fine-tunes a pretrained ResNet-50 on the facial palsy dataset.
Compares against the Random Forest baseline.

Folder structure expected:
    data/
      normal/
        img1.jpg ...
      palsy/
        img1.jpg ...

Usage:
    python cnn_train.py --data_dir ./data

Requirements:
    pip install torch torchvision matplotlib scikit-learn tqdm
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve)

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-4
NUM_CLASSES = 2

print(f"[INFO] Using device: {DEVICE}")
if DEVICE.type == "cpu":
    print("[INFO] No GPU found — training on CPU (will be slower but works fine)")


# ── Data transforms ───────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225]),   # ImageNet std
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Dataset loading ───────────────────────────────────────────────────────────
def load_datasets(data_dir):
    full_dataset = datasets.ImageFolder(data_dir)
    
    # Get person ID from filename (number before first underscore)
    # Letters = random photos, numbers = person ID
    def get_person_id(path):
        fname = os.path.basename(path)
        prefix = fname.split('_')[0]
        return prefix if prefix.isdigit() else fname  # letters treated as unique
    
    # Group indices by person
    from collections import defaultdict
    person_indices = defaultdict(list)
    for idx, (path, label) in enumerate(full_dataset.samples):
        pid = get_person_id(path)
        person_indices[pid].append(idx)
    
    # Split persons into train/val
    persons = list(person_indices.keys())
    np.random.seed(42)
    np.random.shuffle(persons)
    n_val_persons = max(1, int(0.2 * len(persons)))
    val_persons   = set(persons[:n_val_persons])
    train_persons = set(persons[n_val_persons:])
    
    train_indices = [i for p in train_persons for i in person_indices[p]]
    val_indices   = [i for p in val_persons   for i in person_indices[p]]
    
    print(f"\n[INFO] Person-based split:")
    print(f"       Train persons: {len(train_persons)}  Val persons: {len(val_persons)}")
    print(f"       Train images : {len(train_indices)}  Val images : {len(val_indices)}")
    print(f"       Val person IDs: {val_persons}")

    train_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(data_dir, transform=val_transforms)

    train_ds = torch.utils.data.Subset(train_dataset, train_indices)
    val_ds   = torch.utils.data.Subset(val_dataset,   val_indices)

    train_targets = [full_dataset.targets[i] for i in train_indices]
    class_counts  = np.bincount(train_targets)
    weights       = 1.0 / class_counts
    sample_weights = [weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, full_dataset.classes


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last block (layer4) for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final classifier
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    return model.to(DEVICE)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc, all_labels, all_probs


# ── Main ──────────────────────────────────────────────────────────────────────
def train(data_dir):
    train_loader, val_loader, classes = load_datasets(data_dir)

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  [], "val_auc": []}

    best_auc   = 0
    best_epoch = 0

    print(f"\n[INFO] Training ResNet-50 for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc, val_labels, val_probs = val_epoch(
            model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f}")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc*100:.1f}%")
        print(f"  Val   Loss: {val_loss:.4f}  Val Acc  : {val_acc*100:.1f}%  AUC: {val_auc:.3f}")
        print(f"  {'✓ Good' if val_acc > 0.8 else '⚠ Still learning'}  |  Gap: {(train_acc - val_acc)*100:.1f}% {'(possible overfit)' if train_acc - val_acc > 0.1 else ''}")

        # Save best model
        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "cnn_model.pth")
            print(f"  ✓ New best model saved (AUC: {best_auc:.3f})")

        print()

    print(f"[INFO] Best model at epoch {best_epoch} with AUC: {best_auc:.3f}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  FINAL TEST RESULTS (best model)")
    print("="*50)

    model.load_state_dict(torch.load("cnn_model.pth"))
    _, _, _, y_true, y_prob = val_epoch(model, val_loader, criterion)
    y_pred = (np.array(y_prob) > 0.5).astype(int)

    print(classification_report(y_true, y_pred, target_names=classes))
    print(f"  ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")

    # ── Comparison with baseline ──────────────────────────────────────────────
    print("\n" + "="*50)
    print("  MODEL COMPARISON")
    print("="*50)
    print(f"  Random Forest (landmarks) : Acc 88%  AUC 0.937")
    print(f"  ResNet-50 CNN (raw images): Acc {val_acc*100:.0f}%  AUC {best_auc:.3f}")
    print("="*50)

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Loss curve
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy curve
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"],   label="Val Acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[2].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    axes[2].plot([0,1],[0,1],"--", color="gray")
    axes[2].set_title("ROC Curve")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("cnn_results.png", dpi=150)
    print("\n[INFO] Saved plots → cnn_results.png")
    print("[INFO] Saved model  → cnn_model.pth")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 for stroke detection")
    parser.add_argument("--data_dir", default="./data",
                        help="Path to data folder with normal/ and palsy/ subfolders")
    args = parser.parse_args()
    train(args.data_dir)
