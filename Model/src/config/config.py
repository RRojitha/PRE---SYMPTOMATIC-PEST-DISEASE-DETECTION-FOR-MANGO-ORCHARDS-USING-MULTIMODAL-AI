"""
Project Configuration

Author : Mithun
Project : Mango Leaf Disease Classification

This file stores all configurable parameters used throughout the project.
No values should be hardcoded in other modules.
"""

from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
CONFUSION_MATRIX_DIR = OUTPUT_DIR / "confusion_matrix"

# Automatically create output folders

for directory in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    REPORT_DIR,
    LOG_DIR,
    GRADCAM_DIR,
    CONFUSION_MATRIX_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Dataset
# ============================================================

IMAGE_SIZE = (224, 224)

CHANNELS = 3

NUM_CLASSES = 8

CLASS_NAMES = [
    "Anthracnose",
    "Bacterial_Canker",
    "Cutting_Weevil",
    "Die_Back",
    "Gall_Midge",
    "Healthy",
    "Powdery_Mildew",
    "Sooty_Mould",
]


# ============================================================
# Training
# ============================================================

BATCH_SIZE = 16          # Safe for MX450

SEED = 42

HEAD_EPOCHS = 12

FINETUNE_EPOCHS = 30

HEAD_LR = 1e-3

FINETUNE_LR = 1e-5

DROPOUT_1 = 0.40

DROPOUT_2 = 0.30

DENSE_UNITS = 256


# ============================================================
# Model
# ============================================================

BACKBONE = "EfficientNetB0"

PRETRAINED = "imagenet"

MODEL_NAME = "mango_leaf_classifier.keras"

CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# Split
# ============================================================

TRAIN_SPLIT = 0.70

VAL_SPLIT = 0.15

TEST_SPLIT = 0.15


# ============================================================
# Augmentation
# ============================================================

ROTATION = 20

ZOOM = 0.10

WIDTH_SHIFT = 0.10

HEIGHT_SHIFT = 0.10

BRIGHTNESS = 0.10


# ============================================================
# Prediction Labels
# ============================================================

HEALTHY_CLASS = "Healthy"


# ============================================================
# Fusion Model (Image + Environmental Sensors)
# ============================================================
# Additive section for the multimodal fusion model. Nothing above this
# point is used differently by the original image-only pipeline.

# Environmental CSV lives outside Model/, at the project root.
ENV_CSV_PATH = PROJECT_ROOT.parent / "mango_disease_environmental_dataset.csv"

# Columns in the CSV that feed the environmental MLP branch, in order.
ENV_FEATURE_COLS = ["temperature_c", "humidity_percent", "wind_speed_kmh"]

# CSV uses "mango_leaf_class" with spaces (e.g. "Die Back"); the dataset
# folders / CLASS_NAMES use underscores (e.g. "Die_Back"). This maps
# CSV label -> CLASS_NAMES label.
ENV_CLASS_NAME_MAP = {name.replace("_", " "): name for name in CLASS_NAMES}

# Once real per-image sensor readings exist, add an "image_filename" column
# to a CSV and set this to that column name to switch from class-level
# sampling to exact per-image pairing (see src/data/env_lookup.py).
ENV_IMAGE_FILENAME_COL = None

FUSION_MODEL_NAME = "mango_fusion_classifier.keras"

ENV_EMBED_UNITS = 16

FUSION_DENSE_UNITS = 128

FUSION_DROPOUT = 0.30

FUSION_HEAD_EPOCHS = 15

FUSION_FINETUNE_EPOCHS = 15

FUSION_HEAD_LR = 1e-3

FUSION_FINETUNE_LR = 1e-5