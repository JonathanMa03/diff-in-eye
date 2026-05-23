from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "retinal-oct-c8" / "RetinalOCT_Dataset" / "RetinalOCT_Dataset"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURE_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

TARGET_SIZE = (256, 128)  # width, height
STRIPE_HEIGHT = 24

SEED = 123
BATCH_SIZE = 8

T = 1000
BETA_START = 1e-4
BETA_END = 2e-2