# Stroke Detection using Computer Vision

Detects facial drooping, a key sign of stroke, using a webcam and machine learning.

## How it works
Uses MediaPipe to extract 18 facial landmark features (mouth asymmetry, eye asymmetry, brow asymmetry) and a Random Forest classifier to predict stroke risk in real time.

## Results
- Accuracy: 88%
- ROC-AUC: 0.937
- Palsy Recall: 84%

## Based on the FAST method
- Face drooping -> detected by facial asymmetry features
- Arm weakness -> pose landmarks
- Speech slurred -> future work
- Time to call -> model triggers alert above 70% risk

## How to run
pip install mediapipe opencv-python pandas scikit-learn joblib

python extract_from_images.py --data_dir ./data --output landmarks.csv

python train_model.py --data landmarks_face_only.csv

python webcam_demo.py

## Limitations
- Trained on facial palsy data, not direct stroke data
- Screening tool only, not a medical diagnosis
- Requires frontal face view
