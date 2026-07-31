# Multi-Disease Chest X-ray Classifier

A CNN-based image classifier that detects **5 classes** from chest X-rays:
`Normal`, `Tuberculosis (TB)`, `Pneumonia`, `Pneumothorax`, and `COPD`
(using the `Emphysema` label as the standard public proxy for COPD, since no
dedicated public COPD-labeled X-ray dataset exists).

Built and tuned to run on modest consumer hardware (tested config: i5-10300H,
GTX 1650 4GB, 8GB RAM).

---

## 1. Overview

This project has three stages, each with its own notebook:

| Stage | Notebook | What it does |
|---|---|---|
| 1 | `tb_data_preprocessing.ipynb` | Early version — merges Montgomery + Shenzhen only (TB vs. Normal) |
| 2 | `multi_disease_data_preprocessing.ipynb` | **Main preprocessing** — merges all 3 datasets into 5 classes |
| 3 | `model_training.ipynb` | Trains a MobileNetV2 transfer-learning model on the preprocessed data |

If you're doing the full 5-class project, use notebooks **2 and 3**. Notebook 1
is kept only as a simpler TB-only reference/starting point.

---

## 2. Why these diseases (and why not Asthma or COVID)

The original target list was Tuberculosis, COPD, Asthma, and Pneumonia. Asthma
was dropped because it has no reliable, distinct radiographic signature on a
chest X-ray (it's diagnosed clinically/via spirometry, not imaging) — so no
meaningful public dataset or trained model exists for detecting it from X-rays.
Pneumothorax was added in its place because it has a strong, well-documented
radiographic signature and is commonly paired with TB/Pneumonia/COPD in
published research.

---

## 3. Datasets used

All three datasets are public and directly downloadable — no private/in-house
data is required.

| Dataset | Contributes | Link |
|---|---|---|
| Montgomery County CXR Set | TB, Normal | https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery |
| Shenzhen Hospital CXR Set | TB, Normal | https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen |
| NIH ChestX-ray Sample (5,606 images) | Pneumonia, Pneumothorax, COPD (Emphysema), Normal | https://www.kaggle.com/datasets/nih-chest-xrays/sample |

**Known limitation:** the NIH *sample* dataset is only a 5% random slice of the
full 112K-image NIH ChestX-ray14 dataset, so Pneumonia/Pneumothorax/COPD counts
in it are naturally small (expect roughly 60-260 images per class). The
preprocessing notebook flags this explicitly so you know before training
whether you need to top up any class from the full NIH dataset or a source
like the RSNA Pneumonia Detection Challenge dataset.

---

## 4. Folder structure

Set your project up like this before running anything:

```
tb_project/
├── multi_disease_data_preprocessing.ipynb
├── model_training.ipynb
├── data/
│   ├── montgomery/        <- unzip Montgomery dataset here (CSV/clinical files optional, unused)
│   ├── shenzhen/           <- unzip Shenzhen dataset here (CSV/clinical files optional, unused)
│   └── nih_sample/         <- unzip NIH sample dataset here (sample_labels.csv REQUIRED here)
└── multi_disease_dataset/  <- created automatically by preprocessing notebook
    ├── train/
    │   ├── Normal/ TB/ Pneumonia/ Pneumothorax/ COPD/
    ├── val/
    │   ├── Normal/ TB/ Pneumonia/ Pneumothorax/ COPD/
    └── test/
        ├── Normal/ TB/ Pneumonia/ Pneumothorax/ COPD/
```

**Notes:**
- Exact internal nesting inside `montgomery/`, `shenzhen/`, `nih_sample/`
  doesn't matter — the preprocessing notebook searches recursively for images
  and CSVs.
- `sample_labels.csv` **must** be present somewhere inside `nih_sample/` — this
  is the only CSV actually used by the code (Montgomery/Shenzhen labels come
  from filenames, not CSVs).

---

## 5. Setup

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn pillow nbformat --break-system-packages
```

Requires Python 3.9+ and a CUDA-capable GPU (optional but strongly recommended
— CPU training will be significantly slower).

---

## 6. Step 1 — Data Preprocessing (`multi_disease_data_preprocessing.ipynb`)

**What it does:**
1. Recursively scans Montgomery + Shenzhen for images, extracting TB/Normal
   labels directly from filenames (`_0` = Normal, `_1` = TB)
2. Reads NIH's `sample_labels.csv`, keeping only **single-label** images
   matching Pneumonia, Pneumothorax, Emphysema→COPD, or No Finding→Normal
   (multi-label images are skipped to keep this a clean multi-class problem)
3. Merges all sources into one table, capping total Normal images (default
   1,200) so it doesn't dominate the other classes
4. Prints a class-balance check and flags any class under 100 images
5. Stratified 70/15/15 train/val/test split
6. Resizes every image to 224×224, converts to RGB, and writes them into
   `multi_disease_dataset/train|val|test/<class>/` folders

**Before running:** edit the path variables in the "Set your paths" cell:
```python
MONTGOMERY_DIR = "./data/montgomery"
SHENZHEN_DIR   = "./data/shenzhen"
NIH_SAMPLE_DIR = "./data/nih_sample"
OUTPUT_DIR     = "./multi_disease_dataset"
```

**Output:** a clean, balanced-as-possible 5-class image folder structure, ready
for `image_dataset_from_directory()`.

---

## 7. Step 2 — Model Training (`model_training.ipynb`)

**Architecture:** MobileNetV2 (pretrained on ImageNet, frozen base) + custom
head — chosen specifically because it's lightweight enough to train
comfortably on a 4GB GPU, unlike heavier options like VGG16 or ResNet50.

**What it does:**
1. Configures GPU memory growth (prevents OOM crashes on small VRAM cards)
2. Loads train/val/test sets from the preprocessed folders
3. Computes class weights automatically to counteract the class imbalance
   flagged during preprocessing
4. Applies data augmentation (flip, rotation, zoom, contrast) to training data
   only
5. Builds the model: MobileNetV2 → GlobalAveragePooling → Dropout → Dense(128)
   → Dropout → Dense(5, softmax)
6. Trains with early stopping, checkpointing, and learning-rate reduction on
   plateau
7. **Optional Stage 2:** fine-tunes the last 20 layers of MobileNetV2 at a much
   lower learning rate for an extra accuracy boost
8. Evaluates on the held-out test set: accuracy, AUC, per-class precision/
   recall/F1 (classification report), and a confusion matrix heatmap
9. Saves the best model as `best_model.keras`

**Key settings to know:**
- `BATCH_SIZE = 16` — drop to 8 if you hit `ResourceExhaustedError` (CUDA OOM)
- `EPOCHS_FROZEN = 20`, `EPOCHS_FINETUNE = 10` — early stopping usually cuts
  these shorter in practice
- Expected training time on a GTX 1650: roughly 1-3 minutes/epoch for Stage 1

---

## 8. Interpreting your results

- **Low recall on Pneumonia/Pneumothorax/COPD specifically:** expected if
  those classes ended up with few images after preprocessing — this is a data
  quantity issue, not an architecture problem. Fix by sourcing more images for
  just those classes (targeted NIH ChestX-ray14 pull, or RSNA Pneumonia
  dataset), not by changing the model.
- **High train accuracy, much lower val accuracy:** overfitting — increase
  dropout, add more augmentation, or reduce `EPOCHS_FROZEN`.
- **Both train and val accuracy low:** try Stage 2 fine-tuning, or increase
  epochs.
- **CUDA OOM errors:** lower `BATCH_SIZE` to 8; skip Stage 2 fine-tuning if it
  still fails there.

---

## 9. Loading the trained model later

```python
from tensorflow import keras
model = keras.models.load_model("best_model.keras")
```

---

## 10. Known limitations

- COPD is represented via the `Emphysema` NIH label, not a dedicated COPD
  diagnosis label (none exists publicly for chest X-rays)
- Asthma is excluded — not reliably diagnosable from chest X-ray imaging alone
- NIH sample dataset provides limited counts for the rarer classes; results
  for Pneumonia/Pneumothorax/COPD should be interpreted with that in mind
- All training data is drawn from single-institution or curated public
  datasets — real-world generalization to different X-ray machines,
  populations, or imaging protocols is not guaranteed and would need external
  validation before any clinical use
