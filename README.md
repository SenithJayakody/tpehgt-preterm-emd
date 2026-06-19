# TPEHGT Preterm EHG EMD Analysis

Code for reproducing the empirical mode decomposition (EMD) based electrohysterography (EHG) preterm birth prediction experiments reported in the manuscript.

The pipeline uses the Term-Preterm EHG DataSet with Tocogram (TPEHGT) and evaluates record-wise preterm birth classification from dataset-provided filtered EHG channels.

## Repository Contents

- `config.py` - analysis parameters, paths, feature settings, and cross-validation settings.
- `io_readers.py` - TPEHGT WFDB loading, label parsing, contraction intervals, and fixed-window segmentation.
- `features.py` - EMD decomposition and feature extraction.
- `extract_features.py` - generates feature tables for contraction and fixed 3-minute analyses.
- `classify_groupwise_cv.py` - repeated record-wise grouped cross-validation and metric summaries.
- `requirements.txt` - Python dependencies.

## Data

The TPEHGT WFDB files are not included in this repository.

Download the TPEHGT dataset from PhysioNet and place the WFDB records in:

```text
data/tpehgt/
```

The folder should contain files such as:

```text
tpehgt_p001.dat
tpehgt_p001.hea
tpehgt_p001.atr
tpehgt_t001.dat
tpehgt_t001.hea
tpehgt_t001.atr
```

## Python Setup

Recommended on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce Results

Run the scripts from the repository root in this order:

```powershell
python io_readers.py
python extract_features.py
python classify_groupwise_cv.py
```

Generated files are written to:

```text
outputs/
```

## Main Analysis Settings

The manuscript pipeline uses:

- Filtered EHG channels from the TPEHGT dataset: channels 1, 3, and 5.
- No additional filtering before EMD.
- Fixed non-overlapping 3-minute windows and contraction-annotation windows.
- First four IMFs evaluated separately.
- Fourteen features per EHG channel.
- Repeated 5-fold stratified grouped cross-validation.
- Record-level grouping to prevent segments from the same recording appearing in both train and validation folds.
- Maximum segment probability aggregation for record-level prediction.
- MCC-based threshold selection on training folds.

## Citation

If you use this code, please cite the associated manuscript and the original TPEHGT dataset.
