# Stroke Detection

Detects facial drooping, a key sign of stroke, using a webcam and ML/DL.

## How it works
Uses MediaPipe to extract 18 facial landmark features (mouth asymmetry, eye asymmetry, brow asymmetry) and a Random Forest classifier to predict stroke risk in real time.

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
  
