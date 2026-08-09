from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATASET_DIR = Path("data/tpehgt/1.0.0")
OUTPUT_DIR = Path("outputs")
# OUTPUT_DIR = Path("outputs_raw_original")
FEATURE_DIR = OUTPUT_DIR / "features"
RESULT_DIR = OUTPUT_DIR / "results"
PLOT_DIR = OUTPUT_DIR / "plots"
SIGNAL_PLOT_DIR = OUTPUT_DIR / "signal_plots"

# ---------------------------------------------------------------------
# Dataset and signal settings
# ---------------------------------------------------------------------
FS = 20.0

# TPEHGT signal layout:
# 0 = original EHG1
# 1 = filtered EHG1, 0.08--5.0 Hz
# 2 = original EHG2
# 3 = filtered EHG2, 0.08--5.0 Hz
# 4 = original EHG3
# 5 = filtered EHG3, 0.08--5.0 Hz
# 6 = original TOCO
# 7 = filtered TOCO
#
# Main paper pipeline: use dataset-provided filtered EHG channels before EMD.
EHG_CHANNELS = [1, 3, 5]
EHG_CHANNEL_NAMES = ["ehg1", "ehg2", "ehg3"]
EHG_CHANNEL_DISPLAY_NAMES = {
    "ehg1": "EHG1",
    "ehg2": "EHG2",
    "ehg3": "EHG3",
}
SIGNAL_VERSION = "filtered_0p08_5hz"

# EHG_CHANNELS = [0, 2, 4]
# EHG_CHANNEL_NAMES = ["ehg1", "ehg2", "ehg3"]
# SIGNAL_VERSION = "raw_original"

# ---------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------
FIXED_WINDOW_SEC = 180

# ---------------------------------------------------------------------
# EMD / IMF settings
# ---------------------------------------------------------------------
MAX_IMFS = 4

# Manuscript-friendly numbering: IMF1 means the first IMF returned by EMD.
FINAL_IMF_NUMBER = 1
IMF_SELECTION_NUMBERS = [1, 2, 3, 4]

# Internal Python indexes. FINAL_IMF_INDEX=0 means IMF1.
FINAL_IMF_INDEX = 0
IMF_SELECTION_INDEXES = [0, 1, 2, 3]

# ---------------------------------------------------------------------
# Feature extraction parameters
# ---------------------------------------------------------------------
PEAK_MODE = "abs"
THK = 2.8
MD_SEC = 0.30
WIDTH_REL_HEIGHT = 0.5

BT_CONTRACTION_SEC = 1.8
BT_FIXED_SEC = 2.0

SHANNON_BINS = 50

SAMPEN_M = 2
SAMPEN_R = 0.10
SAMPEN_TAU = 1

PERM_M = 3
PERM_TAU = 1
PERM_NORMALIZE = True

# ---------------------------------------------------------------------
# Record-wise classification
# ---------------------------------------------------------------------
N_SPLITS = 5
N_REPEATS = 30
RANDOM_SEED = 42
RECORD_AGGREGATION = "max"
THRESHOLD_METRIC = "mcc"

# Runtime controls. ``-1`` uses all available CPUs. Estimators that expose
# their own threading controls are kept at one internal worker so this outer
# job-level parallelism does not oversubscribe the machine.
N_JOBS = -1

# Optional development overrides. Keep these as ``None`` for the complete
# manuscript analysis (all experiments, all models, 30 repetitions). They can
# also be overridden from the classification command line.
DEBUG_N_REPEATS = None
DEBUG_EXPERIMENTS = None
DEBUG_MODELS = None

# Completed model/repeat jobs are stored atomically and reused after an
# interrupted classification run. Set to False to disable resume by default;
# the command line also provides --no-resume.
CLASSIFICATION_RESUME = True

# ---------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------
FIG_DPI = 300

# None = generate signal plots for all segments.
# Use 1 or 2 while testing to avoid many figure files.
MAX_SIGNAL_SEGMENTS_PER_RECORD_MODE = None

# Main plots compare top models only to keep figures readable.
TOP_MODELS_FOR_ROC = ["CatBoost", "Random Forest", "Gradient Boosting", "MLP", "SVM"]
