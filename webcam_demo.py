"""
Stroke Detection - Live Webcam Demo
=====================================
Runs your trained model on a live webcam feed.
Shows facial landmarks, asymmetry scores, and stroke risk alert.
"""

import time

import cv2
import numpy as np
import joblib
from mediapipe.tasks.python import vision

from face_features import (
    create_face_landmarker,
    to_mp_image,
    facial_feature_vector,
    FACE_FEATURE_NAMES,
)

# ── Load model & scaler ───────────────────────────────────────────────────────
model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

N_FEATURES = len(FACE_FEATURE_NAMES)

if getattr(scaler, "n_features_in_", N_FEATURES) != N_FEATURES:
    raise SystemExit(
        f"[ERROR] model.pkl/scaler.pkl expect {scaler.n_features_in_} features, "
        f"but this demo only extracts {N_FEATURES} face-only features.\n"
        f"         Train on landmarks_face_only.csv (see fix_data.py / "
        f"train_model.py --data), not the raw landmarks.csv with pose columns."
    )


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_alert(frame, prob):
    h, w = frame.shape[:2]

    if prob > 0.7:
        # Red flashing border
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 20)
        cv2.putText(frame, "STROKE RISK DETECTED", (w//2 - 280, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, "CALL EMERGENCY SERVICES", (w//2 - 240, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
    elif prob > 0.4:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 165, 255), 10)
        cv2.putText(frame, "MONITORING...", (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 165, 255), 2)
    else:
        cv2.putText(frame, "NORMAL", (20, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 200, 0), 2)


def draw_hud(frame, prob, features):
    h, w = frame.shape[:2]

    # Risk bar background
    bar_x, bar_y, bar_w, bar_h = 20, h - 80, 300, 24
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)

    # Risk bar fill
    fill = int(bar_w * prob)
    color = (0, 200, 0) if prob < 0.4 else (0, 165, 255) if prob < 0.7 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (200, 200, 200), 1)
    cv2.putText(frame, f"Stroke Risk: {prob*100:.0f}%",
                (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Feature readout
    if features is not None:
        idx = {name: i for i, name in enumerate(FACE_FEATURE_NAMES)}
        lines = [
            f"Mouth droop : {features[0, idx['mouth_droop_asymmetry']]:.1f}px",
            f"Eye asymm   : {features[0, idx['eye_height_asymmetry']]:.1f}px",
            f"Brow asymm  : {features[0, idx['brow_height_asymmetry']]:.1f}px",
            f"Symmetry    : {features[0, idx['face_symmetry_score']]:.1f}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (w - 280, 80 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Instructions
    cv2.putText(frame, "Press Q to quit", (w - 180, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check your camera is connected.")
        return

    print("[INFO] Webcam demo running. Press Q to quit.")
    print("[INFO] Risk threshold: >70% = alert, 40-70% = monitoring, <40% = normal\n")

    # Smooth predictions over last N frames
    prob_history = []
    SMOOTH_N = 8

    landmarker = create_face_landmarker(video_mode=True, min_confidence=0.5)
    start = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror effect
            img_h, img_w = frame.shape[:2]
            mp_img = to_mp_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_img, timestamp_ms)

            features = None
            prob = prob_history[-1] if prob_history else 0.0

            if result.face_landmarks:
                face_lm = result.face_landmarks[0]

                vision.drawing_utils.draw_landmarks(
                    frame, face_lm,
                    vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=vision.drawing_utils.DrawingSpec(
                        color=(0, 200, 100), thickness=1, circle_radius=1))

                features = facial_feature_vector(face_lm, img_w, img_h)
                prob_raw = model.predict_proba(scaler.transform(features))[0][1]

                prob_history.append(prob_raw)
                if len(prob_history) > SMOOTH_N:
                    prob_history.pop(0)
                prob = float(np.mean(prob_history))

            else:
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

            draw_alert(frame, prob)
            draw_hud(frame, prob, features)

            cv2.imshow("Stroke Detection Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
    print("[INFO] Demo closed.")


if __name__ == "__main__":
    run()
