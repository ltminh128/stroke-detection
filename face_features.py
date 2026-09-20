"""
Stroke Detection - Shared landmark extraction (MediaPipe Tasks API)
===================================================================
MediaPipe removed the legacy `mp.solutions` API in 0.10.31+/1.x, so both the
image extractor and the webcam demo build their landmarkers from here.

Feature order is the contract between training and inference: the CSV columns
written by extract_from_images.py must line up with the vector webcam_demo.py
feeds the scaler. FACE_FEATURE_NAMES is that single source of truth.
"""

import os
import urllib.request

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "face": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task",
    ),
    "pose": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    ),
}

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
    "left_shoulder":  vision.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": vision.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow":     vision.PoseLandmark.LEFT_ELBOW,
    "right_elbow":    vision.PoseLandmark.RIGHT_ELBOW,
    "left_wrist":     vision.PoseLandmark.LEFT_WRIST,
    "right_wrist":    vision.PoseLandmark.RIGHT_WRIST,
}

FACE_FEATURE_NAMES = [
    "mouth_offset_x",
    "mouth_droop_asymmetry",
    "eye_height_asymmetry",
    "brow_height_asymmetry",
    "jaw_tilt",
    "face_symmetry_score",
    "mouth_width",
    "mouth_droop_norm",
    "eye_asymmetry_norm",
    "brow_asymmetry_norm",
]


# ── Model files ───────────────────────────────────────────────────────────────

def ensure_model(kind):
    filename, url = MODELS[kind]
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"[INFO] Downloading {kind} model → {path}")
        urllib.request.urlretrieve(url, path)
    return path


def create_face_landmarker(video_mode=False, min_confidence=0.4):
    mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("face")),
            running_mode=mode,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
    )


def create_pose_landmarker(video_mode=False, min_confidence=0.4):
    mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model("pose")),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
    )


def to_mp_image(rgb):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


# ── Feature extraction ────────────────────────────────────────────────────────

def compute_facial_features(landmarks, img_w, img_h):
    def get_pt(idx):
        lm = landmarks[idx]
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
        abs(features["jaw_tilt"])         * 0.1
    )

    # Mouth width doubles as the normalisation reference
    features["mouth_width"] = abs(pts["mouth_right"][0] - pts["mouth_left"][0])

    mw = features["mouth_width"] + 1e-6
    features["mouth_droop_norm"]    = features["mouth_droop_asymmetry"] / mw
    features["eye_asymmetry_norm"]  = features["eye_height_asymmetry"]  / mw
    features["brow_asymmetry_norm"] = features["brow_height_asymmetry"] / mw

    return features


def facial_feature_vector(landmarks, img_w, img_h):
    feats = compute_facial_features(landmarks, img_w, img_h)
    return np.array([feats[name] for name in FACE_FEATURE_NAMES]).reshape(1, -1)


def compute_pose_features(landmarks, img_w, img_h):
    def get_pt(lm_enum):
        lm = landmarks[int(lm_enum)]
        return np.array([lm.x * img_w, lm.y * img_h, lm.visibility or 0.0])

    pts = {name: get_pt(lm) for name, lm in POSE_LANDMARKS.items()}
    features = {}

    left_wrist_rel  = pts["left_wrist"][1]  - pts["left_shoulder"][1]
    right_wrist_rel = pts["right_wrist"][1] - pts["right_shoulder"][1]
    features["left_wrist_height"]    = left_wrist_rel
    features["right_wrist_height"]   = right_wrist_rel
    features["arm_height_asymmetry"] = abs(left_wrist_rel - right_wrist_rel)

    left_arm  = pts["left_elbow"][:2]  - pts["left_shoulder"][:2]
    right_arm = pts["right_elbow"][:2] - pts["right_shoulder"][:2]
    features["left_arm_angle"]      = float(np.degrees(np.arctan2(*left_arm[::-1])))
    features["right_arm_angle"]     = float(np.degrees(np.arctan2(*right_arm[::-1])))
    features["arm_angle_asymmetry"] = abs(features["left_arm_angle"] - features["right_arm_angle"])
    features["shoulder_tilt"]       = pts["left_shoulder"][1] - pts["right_shoulder"][1]
    features["left_wrist_visibility"]  = float(pts["left_wrist"][2])
    features["right_wrist_visibility"] = float(pts["right_wrist"][2])

    return features
