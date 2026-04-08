"""
Stroke Detection - Live Webcam Demo
=====================================
Runs your trained model on a live webcam feed.
Shows facial landmarks, asymmetry scores, and stroke risk alert.
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib

# ── Load model & scaler ───────────────────────────────────────────────────────
model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils

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

# ── Feature extraction (face only — matches landmarks_face_only.csv) ──────────
def extract_features(face_landmarks, img_w, img_h):
    def get_pt(idx):
        lm = face_landmarks.landmark[idx]
        return np.array([lm.x * img_w, lm.y * img_h])

    pts = {name: get_pt(idx) for name, idx in FACE_LANDMARKS.items()}
    f = {}

    mouth_center_x = (pts["mouth_left"][0] + pts["mouth_right"][0]) / 2
    f["mouth_offset_x"]        = mouth_center_x - pts["nose_tip"][0]
    left_drop  = pts["mouth_left"][1]  - pts["nose_tip"][1]
    right_drop = pts["mouth_right"][1] - pts["nose_tip"][1]
    f["mouth_droop_asymmetry"] = abs(left_drop - right_drop)

    left_eye_h  = abs(pts["left_eye_inner"][1]  - pts["left_eye_outer"][1])
    right_eye_h = abs(pts["right_eye_inner"][1] - pts["right_eye_outer"][1])
    f["eye_height_asymmetry"]  = abs(left_eye_h - right_eye_h)

    left_brow_y  = (pts["left_brow_outer"][1]  + pts["left_brow_inner"][1])  / 2
    right_brow_y = (pts["right_brow_inner"][1] + pts["right_brow_outer"][1]) / 2
    f["brow_height_asymmetry"] = abs(left_brow_y - right_brow_y)

    f["jaw_tilt"] = pts["jaw_left"][1] - pts["jaw_right"][1]

    f["face_symmetry_score"] = (
        f["mouth_droop_asymmetry"] * 0.4 +
        f["eye_height_asymmetry"]  * 0.3 +
        f["brow_height_asymmetry"] * 0.2 +
        abs(f["jaw_tilt"])          * 0.1
    )

    mw = abs(pts["mouth_right"][0] - pts["mouth_left"][0]) + 1e-6
    f["mouth_width"]         = mw
    f["mouth_droop_norm"]    = f["mouth_droop_asymmetry"] / mw
    f["eye_asymmetry_norm"]  = f["eye_height_asymmetry"]  / mw
    f["brow_asymmetry_norm"] = f["brow_height_asymmetry"] / mw

    return np.array(list(f.values())).reshape(1, -1)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_alert(frame, prob):
    h, w = frame.shape[:2]

    if prob > 0.7:
        # Red flashing border
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 20)
        cv2.putText(frame, "⚠ STROKE RISK DETECTED", (w//2 - 280, 60),
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
        lines = [
            f"Mouth droop : {features[0,1]:.1f}px",
            f"Eye asymm   : {features[0,2]:.1f}px",
            f"Brow asymm  : {features[0,3]:.1f}px",
            f"Symmetry    : {features[0,4]:.1f}",
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

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror effect
            img_h, img_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            features = None
            prob = prob_history[-1] if prob_history else 0.0

            if results.multi_face_landmarks:
                face_lm = results.multi_face_landmarks[0]

                # Draw face mesh
                mp_drawing.draw_landmarks(
                    frame, face_lm,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 200, 100), thickness=1, circle_radius=1))

                # Extract & predict
                features = extract_features(face_lm, img_w, img_h)
                features_scaled = scaler.transform(features)
                prob_raw = model.predict_proba(features_scaled)[0][1]

                # Smooth over last N frames
                prob_history.append(prob_raw)
                if len(prob_history) > SMOOTH_N:
                    prob_history.pop(0)
                prob = np.mean(prob_history)

            else:
                cv2.putText(frame, "No face detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

            draw_alert(frame, prob)
            draw_hud(frame, prob, features)

            cv2.imshow("Stroke Detection Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Demo closed.")


if __name__ == "__main__":
    run()