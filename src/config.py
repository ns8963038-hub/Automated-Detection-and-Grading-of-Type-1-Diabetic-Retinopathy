"""Central configuration for the DR detection and grading pipeline.

Every tunable knob lives here so the report's "experimental setup" section can be
written straight from this file.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
IMAGES_RAW = DATA_RAW / "train_images"          # populated by download_data.py
IMAGES_PROCESSED = DATA_PROCESSED / "images"    # populated by preprocess.py
LABELS_CSV = DATA_RAW / "train.csv"
SPLITS_CSV = DATA_PROCESSED / "splits.csv"

OUTPUTS = ROOT / "outputs"
MODEL_DIR = OUTPUTS / "models"
FIGURE_DIR = OUTPUTS / "figures"
LOG_DIR = OUTPUTS / "logs"

for _d in (DATA_PROCESSED, IMAGES_PROCESSED, MODEL_DIR, FIGURE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Task definition -- stage 7 of the system architecture
# --------------------------------------------------------------------------
NUM_CLASSES = 5
CLASS_NAMES = [
    "0 - No DR",
    "1 - Mild",
    "2 - Moderate",
    "3 - Severe",
    "4 - Proliferative DR",
]
# Short labels for plots
CLASS_SHORT = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# --------------------------------------------------------------------------
# Preprocessing -- stage 2
# --------------------------------------------------------------------------
PREPROCESS_SIZE = 456        # size images are cached at on disk
CIRCLE_CROP = True           # remove black border, crop to the retinal circle
BEN_GRAHAM = True            # Gaussian-blur subtraction to boost lesion contrast
BEN_GRAHAM_SIGMA = 10        # sigma as a fraction: sigma = size / BEN_GRAHAM_SIGMA

# ImageNet statistics (pretrained backbones expect these)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------
# Data splitting -- stage 3
# --------------------------------------------------------------------------
VAL_SIZE = 0.15
TEST_SIZE = 0.15   # held out entirely until final evaluation
SEED = 42

# --------------------------------------------------------------------------
# Models -- stages 4/5/6
# --------------------------------------------------------------------------
MODELS = {
    # Baseline: convolutional network trained from scratch. Establishes the
    # "no transfer learning" reference point for the results chapter.
    "baseline": {
        "arch": "baseline_cnn",
        "img_size": 224,
        "batch_size": 32,
        "epochs": 40,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "pretrained": False,
    },
    # Final model: ImageNet-pretrained EfficientNet-B3, fine-tuned end to end.
    "efficientnet": {
        "arch": "efficientnet_b3",
        "img_size": 300,
        "batch_size": 16,
        "epochs": 25,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "pretrained": True,
    },
    # Same backbone at 456px. Microaneurysms are 10-20px in a full-resolution
    # fundus image, so at 300px they are sub-pixel and effectively invisible -
    # which is exactly what separates grade 1 from grade 2. The cached images
    # are already 456px, so nothing needs re-preprocessing.
    "efficientnet456": {
        "arch": "efficientnet_b3",
        "img_size": 456,
        "batch_size": 8,      # 456^2 is 2.3x the pixels of 300^2
        "epochs": 20,
        "lr": 2e-4,           # lower, to suit the smaller batch
        "weight_decay": 1e-5,
        "pretrained": True,
    },
    # Secondary transfer-learning model, useful as a third row in the
    # comparison table.
    "resnet": {
        "arch": "resnet50",
        "img_size": 300,
        "batch_size": 16,
        "epochs": 25,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "pretrained": True,
    },
}
DEFAULT_MODEL = "efficientnet"

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
NUM_WORKERS = 4
LABEL_SMOOTHING = 0.05

# Swappable components, so the report can show a real ablation rather than
# asserting these choices are best. Both are exposed as CLI flags on src.train.
#
#   activation -- the non-linearity INSIDE the network (baseline CNN only;
#                 pretrained backbones ship with their own: EfficientNet uses
#                 SiLU, ResNet uses ReLU, and changing those would invalidate
#                 the ImageNet weights).
#   optimizer  -- the rule that turns gradients INTO weight updates.
#
# These are two different jobs and every run uses one of each.
ACTIVATION = "relu"          # relu | leaky_relu | gelu | silu | elu | mish
OPTIMIZER = "adamw"          # adamw | adam | sgd | rmsprop
SGD_MOMENTUM = 0.9
USE_CLASS_WEIGHTS = True     # APTOS is heavily imbalanced towards grade 0
WARMUP_EPOCHS = 2
EARLY_STOP_PATIENCE = 8
GRAD_CLIP = 1.0
MONITOR_METRIC = "qwk"       # quadratic weighted kappa: the APTOS metric
