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

from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve)

# ── Load & prepare data ───────────────────────────────────────────────────────

def person_id(filename):
    """Person ID = the digits before the first underscore (e.g. '007_smile.jpg').
    Anything else is its own group, which groups nothing — load_data warns loudly
    when that happens, because it silently reintroduces identity leakage."""
    prefix = str(filename).split("_")[0]
    return prefix if prefix.isdigit() else str(filename)


def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in ["image", "label"]]
    X = df[feature_cols].values
    y = (df["label"] == "palsy").astype(int).values  # 1=palsy, 0=normal
    groups = df["image"].map(person_id).values

    print(f"\n[INFO] Dataset: {len(df)} samples, {len(feature_cols)} features")
    print(f"       Normal: {(y==0).sum()}  |  Palsy: {(y==1).sum()}")

    n_groups = len(set(groups))
    print(f"       Persons (groups): {n_groups}  "
          f"({len(df)/n_groups:.1f} images per person on average)")

    if n_groups == len(df):
        print("\n" + "!"*70)
        print("  WARNING: every image became its own group, so grouping does nothing")
        print("  and the scores below are inflated by identity leakage. Filenames")
        print("  must start with the person's digits, e.g. 007_smile.jpg, for the")
        print("  split to keep one person out of both train and test.")
        print("!"*70)

    return X, y, feature_cols, groups


# ── Train & evaluate ──────────────────────────────────────────────────────────

def train(csv_path):
    X, y, feature_cols, groups = load_data(csv_path)

    # Hold out whole people, not random images: with many photos per person a
    # random split puts the same face in train and test, and the model scores
    # well by recognising the individual rather than detecting droop.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train    = groups[train_idx]

    print(f"\n[INFO] Person-based split: "
          f"{len(set(groups_train))} train persons / "
          f"{len(set(groups[test_idx]))} test persons  |  "
          f"{len(train_idx)} train images / {len(test_idx)} test images")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Model 1: Random Forest
    print("\n[1/2] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",   # handles imbalance automatically
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_scores = cross_val_score(rf, X_train, y_train, groups=groups_train,
                                cv=cv, scoring="recall")
    print(f"      CV Recall: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

    # Model 2: MLP Neural Network
    print("\n[2/2] Tuning MLP Neural Network...")
    mlp_params = {
        'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64)],
        'activation':         ['relu', 'tanh'],
        'alpha':              [0.0001, 0.001, 0.01],
        'learning_rate':      ['constant', 'adaptive'],
    }
    mlp_base = MLPClassifier(
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp_grid = GridSearchCV(
        mlp_base, mlp_params,
        cv=cv, scoring='recall',
        n_jobs=-1, verbose=1
    )
    mlp_grid.fit(X_train, y_train, groups=groups_train)
    mlp = mlp_grid.best_estimator_
    mlp_scores = cross_val_score(mlp, X_train, y_train, groups=groups_train,
                                 cv=cv, scoring='recall')

    print(f"      Best params : {mlp_grid.best_params_}")
    print(f"      CV Recall   : {mlp_scores.mean():.3f} ± {mlp_scores.std():.3f}")

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
    parser.add_argument("--data", default="landmarks_face_only.csv",
                        help="Path to landmarks CSV (face-only features, matching "
                             "webcam_demo.py's inputs — run fix_data.py on "
                             "landmarks.csv first if you haven't already)")
    args = parser.parse_args()
    train(args.data)