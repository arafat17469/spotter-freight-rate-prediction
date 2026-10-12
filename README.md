# Freight rate prediction

My solution to the Spotter ML Engineer assessment. The task is to predict
`posted_rate` for 12,000 loads and make a December chart for the fixed
Lexington to Fort Wayne route.

## Approach

The labeled data covers January through October 2025. The prediction data starts
in November, so I used date-based splits rather than a random split. August and
September are the development months; October is held out until model selection
is finished. Imputers and encoders are fitted on the training portion of each split.

I compared a median baseline, Ridge regression and three histogram gradient
boosting settings. The lowest average development MAE selects each model.
The full model uses the supplied market signals and coordinates. The December
file has only six input fields, so it needs a separate model trained on those fields.
Both selected models are then fitted on all labeled rows for the final predictions.

The three tree settings use 300 iterations and a learning rate of 0.06:

| Setting | Maximum leaf nodes | L2 regularization |
| --- | ---: | ---: |
| hgb_1 | 15 | 10 |
| hgb_2 | 31 | 10 |
| hgb_3 | 31 | 50 |

## Findings and results

There are 48,000 labeled loads. Training data has 300 missing weights and 374
missing market index values; numeric fields use median imputation. There are no
duplicate rows across the original predictor columns. The prediction data also
has 736 pickup/delivery pairs absent from training, so unknown categories need
to work at prediction time.

Results from the October holdout:

| Inputs | Selected model | MAE ($) | RMSE ($) | R2 | MAPE (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | hgb_3 | 196.66 | 672.68 | 0.806 | 9.37 |
| Six December fields | hgb_1 | 169.05 | 661.17 | 0.813 | 8.83 |

The core model did better on October, although the full model uses more fields.
The input sets were fixed before testing, so I kept their development-selected
models for their respective tasks. RMSE is much higher than MAE; some large
errors remain, as the actual-versus-predicted plot shows.

These results describe the October holdout. The November/December labels are
not available. The extra two-month check overlaps September development data,
so it is a horizon check rather than another independent test. With only ten
months of training data, the December chart has limited evidence for seasonal
behavior. `quote_signal` is used because it is supplied for prediction; its
availability before the quoted rate would need checking in a production system.

## Run

Use Python 3.10 or newer. In PowerShell, from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe freight_pipeline.py
```

The same code is in `spotter-freight-rate-prediction-ml-assessment.ipynb`.
With the dependencies installed in your Jupyter environment, open it and run
the cells in order. CPU is enough.

The completed run used Python 3.12.10. Its package versions are recorded in
[the output requirements](freight_submission/requirements.txt), and the seed,
input hashes and selected settings are in [run metadata](freight_submission/run_metadata.json).

Put the four assessment CSVs in this folder, `data/`, or `upload/`:

- `train-test.csv`
- `validation.csv`
- `validation-predictions-template.csv`
- `december-chart-inputs.csv`

Filenames with underscores also work. For another folder, set
`$env:SPOTTER_DATA_DIR = 'C:\path\to\data'` before running. On Kaggle, attach
the dataset and run all cells; the code searches `/kaggle/input`.
The supplied datasets are kept out of this repository. No API keys or `.env`
file are needed.

## Files

`freight_pipeline.py` runs training, prediction, scoring and report generation.
`score.py` and `assessment-readme.md` are the original supplied assessment files.
The notebook explains each step alongside the code.

Outputs go to `freight_submission/` beside the script, or under `/kaggle/working`
on Kaggle. The completed run includes:

- [Validation predictions](freight_submission/validation_predictions.csv)
- [Completed December inputs](freight_submission/data/december_chart_inputs.csv)
- [December chart](freight_submission/scorer_results/candidate_december.png)
- [Validation report](freight_submission/Freight_Rate_Validation_Report.pdf)
- [Development metrics](freight_submission/development_metrics.csv) and [October metrics](freight_submission/october_test_metrics.csv)
- [Loom notes](freight_submission/Loom_outline.md)

The pipeline runs the original scorer automatically. To check the saved files again:

```powershell
.\.venv\Scripts\python.exe score.py --predictions freight_submission/validation_predictions.csv --december-predictions freight_submission/data/december_chart_inputs.csv --output-dir freight_submission/scorer_results
```

The scorer checks IDs, columns and prediction values, then creates the December
chart. It does not calculate the hidden-label score.

The run also saves model files, package versions and input hashes locally.
`Spotter_Freight_Submission.zip` contains the handoff files without the model binaries.
The 2-3 minute Loom recording is a separate submission item.
