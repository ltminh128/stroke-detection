# Stroke Detection

Detects facial drooping, a key sign of stroke, using a webcam and ML/DL.

## How it works
Uses MediaPipe to extract facial landmark features (mouth asymmetry, eye asymmetry,
brow asymmetry) and a classifier to predict stroke risk in real time.

## Setup
```bash
pip install -r requirements.txt

python extract_from_images.py --data_dir ./data   # -> landmarks.csv
python fix_data.py                                # -> landmarks_face_only.csv
python train_model.py                             # -> model.pkl, scaler.pkl
python webcam_demo.py                              # live demo
```
`data/` holds one subfolder per class (`data/normal/`, `data/palsy/`). The
MediaPipe `.task` model files download automatically into `models/` on first run.

Built on the MediaPipe Tasks API — the legacy `mp.solutions` API this project
originally used was removed from the pip package in mediapipe 0.10.31+.

## Based on the FAST method
- Face drooping -> detected by facial asymmetry features
- Arm weakness -> pose landmarks
- Time to call -> model triggers alert above 70% risk

## Tuning
- MLP Hyperparameter tuning
- CNN fine-tuning (ResNet-50), with a person-ID-based train/val split to avoid
  the same subject appearing in both sets — relies on filenames encoding
  subject IDs as a leading number (e.g. `123_01.jpg`); verify this holds for
  your dataset before trusting the split
## Results
- Accuracy: 94%
- ROC-AUC: 0.965
- Palsy Recall: 88%

## Model selection
- MLP outperformed on a dataset of ~20,000 facial images

## Limitations
- Trained on facial palsy data, not direct stroke data
- Screening tool only, not a medical diagnosis
- Requires frontal face view

## To-dos:
- Find specific data for CNN (apply for MEEI datasets)
- Tuning ideas:
    + Feature selection
    + Add more facial landmarks
  
