"""
Stroke Detection - Image Folder Landmark Extractor
===================================================
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_pose     = mp.solutions.pose

FACE_LANDMARKS = {
    "mouth_left":       61,
    "mouth_right":      291,
    "mouth_top":        13,
    "mouth_bottom":     14,
    "left_eye_outer":   33,
    "left_eye_inner":   133,
    "right_eye_inner":  362,
    "right_eye_outer":  263,
    "left_brow_outer":  70,
    "left_brow_inner":  107,
    "right_brow_inner": 336,
    "right_brow_outer": 300,
    "nose_tip":         4,
    "jaw_left":         172,
    "jaw_right":        397,
    "chin":             152,
}

POSE_LANDMARKS = {
    "left_shoulder":  mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow":     mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow":    mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist":     mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist":    mp_pose.PoseLandmark.RIGHT_WRIST,
}

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Feature extraction ────────────────────────────────────────────────────────

def compute_facial_features(face_landmarks, img_w, img_h):
    def get_pt(idx):
        lm = face_landmarks.landmark[idx]
        return np.array([lm.x * img_w, lm.y * img_h])

    pts = {name: get_pt(idx) for name, idx in FACE_LANDMARKS.items()}
    features = {}

    # Mouth asymmetry
    mouth_center_x = (pts["mouth_left"][0] + pts["mouth_right"][0]) / 2
    features["mouth_offset_x"] = mouth_center_x - pts["nose_tip"][0]
    left_drop  = pts["mouth_left"][1]  - pts["nose_tip"][1]
    right_drop = pts["mouth_right"][1] - pts["nose_tip"][1]
    features["mouth_droop_asymmetry"] = abs(left_drop - right_drop)

    # Eye asymmetry
    left_eye_h  = abs(pts["left_eye_inner"][1]  - pts["left_eye_outer"][1])
    right_eye_h = abs(pts["right_eye_inner"][1] - pts["right_eye_outer"][1])
    features["eye_height_asymmetry"] = abs(left_eye_h - right_eye_h)

    # Brow asymmetry
    left_brow_y  = (pts["left_brow_outer"][1]  + pts["left_brow_inner"][1])  / 2
    right_brow_y = (pts["right_brow_inner"][1] + pts["right_brow_outer"][1]) / 2
    features["brow_height_asymmetry"] = abs(left_brow_y - right_brow_y)

    # Jaw tilt
    features["jaw_tilt"] = pts["jaw_left"][1] - pts["jaw_right"][1]

    # Overall symmetry score
    features["face_symmetry_score"] = (
        features["mouth_droop_asymmetry"] * 0.4 +
        features["eye_height_asymmetry"]  * 0.3 +
        features["brow_height_asymmetry"] * 0.2 +
        abs(features["jaw_tilt"])          * 0.1
    )

    # Mouth width (normalisation reference)
    features["mouth_width"] = abs(pts["mouth_right"][0] - pts["mouth_left"][0])

    # Normalised versions (divide by mouth width to be scale-invariant)
    mw = features["mouth_width"] + 1e-6
    features["mouth_droop_norm"]  = features["mouth_droop_asymmetry"] / mw
    features["eye_asymmetry_norm"] = features["eye_height_asymmetry"] / mw
    features["brow_asymmetry_norm"] = features["brow_height_asymmetry"] / mw

    return features


def compute_pose_features(pose_landmarks, img_w, img_h):
    def get_pt(lm_enum):
        lm = pose_landmarks.landmark[lm_enum]
        return np.array([lm.x * img_w, lm.y * img_h, lm.visibility])

    pts = {name: get_pt(lm) for name, lm in POSE_LANDMARKS.items()}
    features = {}

    left_wrist_rel  = pts["left_wrist"][1]  - pts["left_shoulder"][1]
    right_wrist_rel = pts["right_wrist"][1] - pts["right_shoulder"][1]
    features["left_wrist_height"]    = left_wrist_rel
    features["right_wrist_height"]   = right_wrist_rel
    features["arm_height_asymmetry"] = abs(left_wrist_rel - right_wrist_rel)

    left_arm  = pts["left_elbow"][:2]  - pts["left_shoulder"][:2]
    right_arm = pts["right_elbow"][:2] - pts["right_shoulder"][:2]
    features["left_arm_angle"]       = float(np.degrees(np.arctan2(*left_arm[::-1])))
    features["right_arm_angle"]      = float(np.degrees(np.arctan2(*right_arm[::-1])))
    features["arm_angle_asymmetry"]  = abs(features["left_arm_angle"] - features["right_arm_angle"])
    features["shoulder_tilt"]        = pts["left_shoulder"][1] - pts["right_shoulder"][1]
    features["left_wrist_visibility"]  = float(pts["left_wrist"][2])
    features["right_wrist_visibility"] = float(pts["right_wrist"][2])

    return features


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

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4) as face_mesh, \
         mp_pose.Pose(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4) as pose:

        for img_path, label in tqdm(image_files, desc="Extracting landmarks"):
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            img_h, img_w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            face_results = face_mesh.process(rgb)
            pose_results = pose.process(rgb)

            row = {"image": os.path.basename(img_path), "label": label}

            if not face_results.multi_face_landmarks:
                # No face detected — skip this image
                skipped += 1
                continue

            face_feats = compute_facial_features(
                face_results.multi_face_landmarks[0], img_w, img_h)
            row.update(face_feats)

            if pose_results.pose_landmarks:
                pose_feats = compute_pose_features(
                    pose_results.pose_landmarks, img_w, img_h)
                row.update(pose_feats)

            rows.append(row)

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