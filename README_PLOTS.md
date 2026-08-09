# Paper figure reproduction

The figure scripts are located in the repository root beside `config.py`,
`features.py`, `io_readers.py`, and `classify_groupwise_cv.py`.

## Files

- `paper_style.py` — shared colors, model names, and manuscript styles.
- `plot_signal_figures.py` — Figs. 2–5 and their underlying CSV values.
- `plot_performance_figures.py` — Figs. 6–7 and their underlying CSV values.
- `plot_feature_distributions.py` — current 14-panel Fig. 8 and its record-level/effect-size CSVs.
- `grouped_permutation_importance.py` — recomputes current RF grouped importance and generates Fig. 9.
- `run_all_with_plots.ps1` — complete analysis plus figure-generation sequence.

All final plots are written to `outputs/plots/paper/` as both PNG (300 dpi) and vector PDF.

## Recommended run order

```bash
python io_readers.py
python extract_features.py
python classify_groupwise_cv.py
python grouped_permutation_importance.py
python plot_signal_figures.py --record <representative_record> --channel <ehg1|ehg2|ehg3> --segment-id <0-9>
python plot_performance_figures.py
python plot_feature_distributions.py
```

Select the representative recording, channel, and segment explicitly for the
final signal figures:

```bash
python plot_signal_figures.py \
  --record tpehgt_p001 \
  --channel ehg2 \
  --segment-id 0
```

Replace these example values with the representative signal selected for the
manuscript. If the arguments are omitted, the script deterministically selects
the first preterm recording, uses EHG2, and uses fixed segment 0; that fallback
should be used for testing rather than an undocumented final selection.

## Important methodological updates reflected here

- Uses `BURST_RATE`, not the superseded raw `BURST_COUNT`.
- Uses **Average Precision (AP)** throughout; it does not report trapezoidal
  precision-recall AUC.
- Reads current short model labels (`RF`, `CB`, `GB`, etc.) written by `classify_groupwise_cv.py`.
- Fig. 6 uses all pooled repeated OOF recording scores. Its pooled OOF ROC-AUC
  and pooled OOF AP describe the plotted curves and are saved under explicit
  `pooled_oof_*` column names. These values are not mathematically identical to
  the mean of the 30 repeat-level ROC-AUC/AP values in `summary_metrics.csv`.
- Fig. 7 uses majority vote over the 30 saved fold-specific binary OOF
  predictions for each recording. It does not apply a score threshold of 0.5.
  Each matrix therefore contains 26 recordings. An exact 15/15 vote raises an
  error and must be resolved explicitly.
- Grouped permutation importance imports the current RF model, recording-level
  split construction, preprocessing pipeline, MAX aggregation, and AP function
  from `classify_groupwise_cv.py`. Within each repetition it pools all five OOF
  folds before calculating baseline and permuted AP on 26 recordings.
- Fig. 2's full-record IMF1 is clearly marked as visualization-only; classification still applies EMD after segmentation.
- Fig. 8 shows descriptive recording-level feature means: channels are averaged
  within each segment, followed by segment averaging within each recording.
  This is distinct from MAX aggregation of classifier scores.
