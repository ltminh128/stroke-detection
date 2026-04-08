"""
Stroke Detection - Model Training Script
========================================
Trains a classifier on the extracted landmark features.
Uses class weights to handle imbalanced data.
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve)

# ── Load & prepare data ───────────────────────────────────────────────────────

def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in ["image", "label"]]
    X = df[feature_cols].values
    y = (df["label"] == "palsy").astype(int).values  # 1=palsy, 0=normal

    print(f"\n[INFO] Dataset: {len(df)} samples, {len(feature_cols)} features")
    print(f"       Normal: {(y==0).sum()}  |  Palsy: {(y==1).sum()}")

    return X, y, feature_cols


# ── Train & evaluate ──────────────────────────────────────────────────────────

def train(csv_path):
    X, y, feature_cols = load_data(csv_path)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Model 1: Random Forest
    print("\n[1/2] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",   # handles imbalance automatically
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="recall")
    print(f"      CV Recall: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

    # Model 2: MLP Neural Network 
    print("\n[2/2] Training MLP Neural Network...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_train, y_train)
    mlp_scores = cross_val_score(mlp, X_train, y_train, cv=5, scoring="recall")
    print(f"      CV Recall: {mlp_scores.mean():.3f} ± {mlp_scores.std():.3f}")

    # Pick best model
    best_name  = "Random Forest" if rf_scores.mean() >= mlp_scores.mean() else "MLP"
    best_model = rf if rf_scores.mean() >= mlp_scores.mean() else mlp
    print(f"\n[INFO] Best model: {best_name}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("\n" + "="*50)
    print("  TEST SET RESULTS")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=["normal", "palsy"]))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print("="*50)

    # Always show Random Forest feature importance regardless of best model
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 most important features (Random Forest):")
    print(importances.round(3).to_string())

    print(f"\nBest model: {best_name}")
    print(f"Random Forest CV Recall: {rf_scores.mean():.3f}")
    print(f"MLP CV Recall:           {mlp_scores.mean():.3f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["normal", "palsy"]).plot(ax=axes[0])
    axes[0].set_title("Confusion Matrix")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
    axes[1].plot([0,1],[0,1],"--", color="gray")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\n[INFO] Saved results plot → results.png")

    # Save model & scaler
    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler,     "scaler.pkl")
    print("[INFO] Saved model  → model.pkl")
    print("[INFO] Saved scaler → scaler.pkl")
    print("\nNext step: run webcam_demo.py to test your model live!")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stroke detection classifier")
    parser.add_argument("--data", default="landmarks.csv", help="Path to landmarks CSV")
    args = parser.parse_args()
    train(args.data)