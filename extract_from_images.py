"""
Stroke Detection - Image Folder Landmark Extractor
===================================================
Walks a folder of labelled images and writes one row of landmark features
per image. Uses the MediaPipe Tasks API (see face_features.py).
"""

import cv2
import pandas as pd
import argparse
import os
from tqdm import tqdm

from face_features import (
    create_face_landmarker,
    create_pose_landmarker,
    to_mp_image,
    compute_facial_features,
    compute_pose_features,
)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Main processing loop ──────────────────────────────────────────────────────

def process_folder(data_dir: str, output_csv: str):
    """
    Walk data_dir looking for subfolders. Each subfolder name = class label.
    E.g. data/normal/ → label "normal", data/palsy/ → label "palsy"
    """
    rows = []
    skipped = 0

    # Collect (image_path, label) pairs
    image_files = []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTS:
                image_files.append((os.path.join(label_dir, fname), label))

    if not image_files:
        print(f"[ERROR] No images found in {data_dir}")
        print("        Make sure your folder structure is: data/label_name/image.jpg")
        return

    print(f"\n[INFO] Found {len(image_files)} images across "
          f"{len(set(lbl for _, lbl in image_files))} classes\n")

    face_landmarker = create_face_landmarker()
    pose_landmarker = create_pose_landmarker()

    try:
        for img_path, label in tqdm(image_files, desc="Extracting landmarks"):
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            img_h, img_w = img.shape[:2]
            mp_img = to_mp_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            face_result = face_landmarker.detect(mp_img)
            if not face_result.face_landmarks:
                # No face detected — skip this image
                skipped += 1
                continue

            row = {"image": os.path.basename(img_path), "label": label}
            row.update(compute_facial_features(
                face_result.face_landmarks[0], img_w, img_h))

            pose_result = pose_landmarker.detect(mp_img)
            if pose_result.pose_landmarks:
                row.update(compute_pose_features(
                    pose_result.pose_landmarks[0], img_w, img_h))

            rows.append(row)
    finally:
        face_landmarker.close()
        pose_landmarker.close()

    if not rows:
        print("[ERROR] No landmarks extracted. Check your images.")
        return

    df = pd.DataFrame(rows)

    # Fill missing pose columns with 0 (images where pose wasn't detected)
    df = df.fillna(0)

    df.to_csv(output_csv, index=False)

    # Summary
    print(f"\n{'='*50}")
    print(f"  Extracted : {len(df)} images")
    print(f"  Skipped   : {skipped} (no face detected or unreadable)")
    print(f"  Features  : {len(df.columns) - 2} per image")
    print(f"  Saved to  : {output_csv}")
    print(f"{'='*50}")
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())
    print("\nFeature preview:")
    print(df.drop(columns=["image", "label"]).describe().round(3))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from an image dataset folder")
    parser.add_argument("--data_dir", required=True,
                        help="Root folder containing class subfolders (e.g. ./data)")
    parser.add_argument("--output",   default="landmarks.csv",
                        help="Output CSV filename (default: landmarks.csv)")
    args = parser.parse_args()

    process_folder(args.data_dir, args.output)
