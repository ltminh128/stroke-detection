# Stroke Detection

Detects facial drooping, a key sign of stroke, from a webcam feed, using
MediaPipe facial landmarks and a classical ML classifier.

> **Status: research prototype, not validated.** The headline numbers this
> project once reported turned out to be measuring the wrong thing (see
> [The overfitting problem](#the-overfitting-problem)). Treat all current
> figures as provisional until re-measured with the person-aware splits.

---

## How it works

There are **two separate models** in this repository. They are independent
experiments and do not feed into each other.

### 1. The landmark pipeline (this is what actually runs)

```
 photo / webcam frame
         │
         ▼
 MediaPipe FaceLandmarker          ← Google's pretrained CNN, frozen.
 (478 landmark points)               Not trained or tuned here.
         │
         ▼
 10 hand-built measurements        ← mouth droop asymmetry, eye asymmetry,
 (face_features.py)                  brow asymmetry, jaw tilt, …
         │
         ▼
 Random Forest  or  MLP            ← THIS is the model that gets trained.
 (train_model.py → model.pkl)         Whichever scores better is kept.
```

`webcam_demo.py` loads `model.pkl` and `scaler.pkl`. This is the deployed path.

Note the only convolutional network involved is MediaPipe's landmarker, which is
a fixed, pretrained feature extractor. The trained classifier is a Random Forest
or a small fully-connected MLP, **not** a CNN.

### 2. The CNN experiment (currently disconnected)

`cnn_train.py` fine-tunes a ResNet-50 on raw images and writes `cnn_model.pth`.
**Nothing else in the repo loads that file.** It is a parallel alternative
approach, not a tuning stage for the landmark model, and it is not part of the
demo. See [the CNN section below](#why-the-cnn-is-especially-vulnerable) before
drawing any conclusion from its scores.

---

## Setup

```bash
pip install -r requirements.txt
```

Run each step in order, waiting for each to finish:

```bash
python extract_from_images.py --data_dir ./data   # -> landmarks.csv       (slow)
python fix_data.py                                # -> landmarks_face_only.csv
python train_model.py                             # -> model.pkl, scaler.pkl
python webcam_demo.py                             # live demo
```

The MediaPipe `.task` model files download automatically into `models/` on first
run. `--data_dir` may point anywhere, including outside the repo.

Built on the **MediaPipe Tasks API**. The legacy `mp.solutions` API this project
originally used was removed from the pip package in mediapipe 0.10.31+, so older
code fails with `AttributeError: module 'mediapipe' has no attribute 'solutions'`.

### Dataset layout

```
data/
  normal/
    007_smile.jpg
    007_rest.jpg
    ...
  palsy/
    012_smile.jpg
    ...
```

**Filenames must begin with the person's ID followed by an underscore.** This is
not cosmetic. It is how the training scripts keep one person out of both train
and test. `train_model.py` parses the digits before the first underscore as the
person ID and prints what it found:

```
Persons (groups): 18  (1100.0 images per person on average)
```

If instead you see a count equal to your image count, plus a loud `!!!!!!`
warning block, the grouping silently did nothing and **every score produced is
invalid**. Fix the filenames before trusting any result.

---

## The overfitting problem

This is the central issue with this project and the reason its early results
were misleading.

### The root cause: the dataset has ~20,000 images but only ~10–20 people

For generalisation, **the effective sample size is the number of distinct
individuals, not the number of photos.** A thousand photos of the same face
carry barely more information about *new* faces than a handful do. Every
technique below is downstream of this single fact.

### What went wrong

`train_model.py` originally split the data randomly over images:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

With many photos per person, the same individual appears in **both** train and
test. The model never has to learn what facial droop looks like. It can learn
"this is person #7, and person #7 is labelled palsy", then recognise person #7
at test time and be scored correct. This is **identity leakage**.

The effect is not subtle. On synthetic data built to mirror this dataset
(20 people × 100 photos, label assigned per person, strong per-person
fingerprint, deliberately weak true signal):

| split | people in both train & test | accuracy | ROC-AUC |
|---|---|---|---|
| random image split (old) | **20** | 1.000 | 1.000 |
| person-grouped split (new) | **0** | 0.500 | 0.500 |

The old split scores *perfectly* on data containing almost no real signal. The
previously reported **94% accuracy / 0.965 ROC-AUC / 88% recall came from this
broken split** and should be disregarded.

### What has been fixed

`train_model.py` now groups every row by person and splits with
`StratifiedGroupKFold`, so no individual spans train and test. Critically, the
5-fold cross-validation is grouped the same way, because that CV selects the
best model and tunes the MLP via `GridSearchCV`, so leaking there picks the
wrong model, not just an optimistic number. Both scripts now warn loudly when
grouping degenerates.

**Expect honest numbers to be substantially lower than 94%.** That is not a
regression; it is the first measurement of the right quantity.

### Why the CNN is especially vulnerable

`cnn_train.py` does implement a person-level split, but several factors make it
far more exposed to this failure than the landmark model. Before trusting any
CNN result, weigh all of these:

1. **Raw pixels make identity trivial to memorise.** The landmark model sees 10
   geometric numbers, which can barely encode *who* someone is. A ResNet sees
   skin texture, hair, glasses, lighting and background. Identity is the
   easiest signal in the image, and gradient descent takes the easiest route.

2. **~15.5M trainable parameters against ~16 training individuals.** Unfreezing
   `layer4` plus the new head leaves 15,489,794 trainable weights. Fitting that
   to roughly a dozen distinct faces is wildly over-parameterised regardless of
   how many photos those faces appear in.

3. **Augmentation multiplies images, not people.** Flips, rotations and colour
   jitter create more views of the *same* individuals. They do nothing about the
   actual bottleneck, while making the training set look reassuringly large.

4. **The validation set is only ~2–4 people.** With a 20% person-level split on
   10–20 individuals, every validation metric rests on a handful of faces.
   Validation AUC will swing violently depending on *which* people land there,
   so small differences between runs or epochs are noise.

5. **Model selection and reporting use the same set.** The best epoch is chosen
   by validation AUC, then that checkpoint's score on *that same validation set*
   is reported as the final result. Selecting the best of 20 epochs against a
   set and then reporting on it is optimistically biased. There is no held-out
   test set of people untouched by selection.

6. **The person-ID split depends entirely on filenames.** Any filename whose
   prefix is not digits becomes its own "person", which quietly disables the
   protection for those files. The script now prints a warning when this
   happens, but it cannot recover identity that filenames do not encode.

7. **Capture-session confounds.** If palsy and normal photos come from different
   sources, cameras, or sessions, the network can score well by learning
   backgrounds, resolution or colour balance. A person-level split does not
   protect against this; only mixed-source data does.

### What would actually help

Ranked by expected impact:

1. **More distinct individuals.** This dominates everything else. Twenty more
   people is worth far more than twenty thousand more photos.
2. **Leave-one-person-out cross-validation.** With so few subjects, a single
   train/test split is high-variance; LOPO-CV uses the data far better and gives
   an error bar.
3. **A held-out test set of people never used for model or epoch selection**,
   kept separate from the validation set.
4. **Prefer the landmark model over the CNN** at this data scale. Hand-built
   geometric features are a strong regulariser precisely because they discard
   identity.
5. **If using the CNN, freeze more of it**: train only the head, or a much
   smaller backbone.

---

## Based on the FAST method

- **F**ace drooping → detected by facial asymmetry features
- **A**rm weakness → pose landmarks (extracted, not yet used by the classifier)
- **T**ime to call → the demo alerts above 70% risk

---

## Results

Earlier figures (94% accuracy, 0.965 ROC-AUC, 88% palsy recall) were produced by
the leaky random split described above and are **not valid**. They are recorded
here only to mark what was superseded.

Re-run `train_model.py` on the person-grouped split to obtain trustworthy
numbers. Report them alongside the number of distinct individuals in train and
test, since that is what determines whether they mean anything.

---

## Limitations

- Trained on facial palsy data, not on stroke patients
- Screening aid only. **Not a medical diagnosis**, and not validated clinically
- Requires a reasonably frontal, well-lit face
- Very few distinct individuals, so per-person generalisation is the binding
  constraint on every result in this repo
- Arm/pose features are extracted but dropped before training by `fix_data.py`,
  so the deployed model uses face features only

---

## To-dos

- Acquire data covering **more individuals** (e.g. apply for the MEEI
  Facial Palsy dataset)
- Switch evaluation to leave-one-person-out CV and report error bars
- Hold out a person-level test set separate from model selection
- Wire the CNN into a fair comparison, or remove it
- Verify palsy and normal images are not separable by capture source
- Tuning ideas: feature selection, additional facial landmarks
