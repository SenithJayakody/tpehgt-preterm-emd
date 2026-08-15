# Publication artifacts

Run `python export_publication_outputs.py` after the complete analysis pipeline.
The exporter places compact CSV artifacts underlying the manuscript tables and
figures in this directory. It reads completed outputs only and fails if any
required source artifact is missing.

Raw PhysioNet data, extracted segment-level feature tables, detailed fold and
segment predictions, fitted models, and classification checkpoints do not
belong in this directory.
