from pathlib import Path
import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

# Number of candidate evidence passages retrieved initially
RETRIEVAL_TOP_K = 10

# Number of passages passed to NLI verification
NLI_TOP_K = 5


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

RANDOM_SEED = 42

# We start small while developing.
# Later we'll scale up for the real experiment.
DEV_SAMPLE_SIZE = 1000


# ============================================================
# GPU SETTINGS
# ============================================================

# Conservative starting batch size for your 8 GB RTX 4060.
EMBEDDING_BATCH_SIZE = 32
NLI_BATCH_SIZE = 16


# ============================================================
# DISPLAY
# ============================================================

print(f"Project root : {PROJECT_ROOT}")
print(f"Device       : {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU          : {torch.cuda.get_device_name(0)}")