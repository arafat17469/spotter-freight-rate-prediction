# Freight rate prediction

Spotter assessment: chronological validation and freight rate predictions.

## Run locally

Use Python 3.10 or newer. From this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe freight_pipeline.py
```

Alternatively, open `spotter-freight-rate-prediction-ml-assessment.ipynb` in a
Jupyter environment with the dependencies installed and run all cells.

The four provided CSVs may stay in this folder, `data/`, or `upload/`:

- `train-test.csv` (also accepts `train_test.csv`)
- `validation.csv`
- `validation-predictions-template.csv` (also accepts `validation_predictions_template.csv`)
- `december-chart-inputs.csv` (also accepts `december_chart_inputs.csv`)

For a different location, set `$env:SPOTTER_DATA_DIR = 'C:\path\to\data'` before
running. That folder must contain all four files. On Kaggle, attach the private
dataset; the notebook detects the folder under `/kaggle/input` automatically.

## Outputs and validation

Outputs are saved in `freight_submission/` beside the script (or under
`/kaggle/working` on Kaggle):

- `validation_predictions.csv`: 12,000 rows with `load_id,predicted_rate`.
- `data/december_chart_inputs.csv`: 31 completed fixed December inputs.
- `scorer_results/candidate_december.png`: chart from the original supplied scorer.
- `Freight_Rate_Validation_Report.pdf`: split approach, measured metrics and chart.
- Metrics CSVs, model artifacts, run metadata, source and Loom outline.

The pipeline runs the employer's scorer automatically. To run it separately:

```powershell
.\.venv\Scripts\python.exe score.py --predictions freight_submission/validation_predictions.csv --december-predictions freight_submission/data/december_chart_inputs.csv --output-dir freight_submission/scorer_results
```

August and September development folds select the model by mean MAE. October is
reserved for the final temporal test. Models then refit on all labeled data.
Validation loads use the full feature set; December chart inputs use a separate
model trained on the six available fields. Hidden validation accuracy is unknown;
the scorer checks file validity and generates the chart.

`Spotter_Freight_Submission.zip` packages the generated handoff files. Keep the
provided data private as described in the notebook. Submit an accessible code
repository, predictions, report, and your own 2–3 minute Loom walkthrough according
to the assessment instructions. The pipeline does not upload or submit anything.
