# Freight rate prediction

These files come from the Spotter assessment pipeline.

To repeat the run, install requirements.txt, put the four original CSVs beside
freight_pipeline.py (or in data/ or upload/), and run `python freight_pipeline.py`.
Set SPOTTER_DATA_DIR for a different input folder. For the notebook, run the
cells in order. On Kaggle, attach the dataset first. CPU is enough.

August and September select the model settings. October is the held-out test.
The final fit uses all labeled rows. Validation loads use the full input set;
the December chart uses the six available fields.

validation_predictions.csv contains the 12,000 final predictions. The completed
December inputs are in data/december_chart_inputs.csv. The original score.py checks
both files and creates scorer_results/candidate_december.png. The PDF explains
the splits and results, and includes that chart. Run details and metrics are
saved alongside it. Hidden validation labels are not available.

The original datasets are not included. The Loom recording is a separate item.
