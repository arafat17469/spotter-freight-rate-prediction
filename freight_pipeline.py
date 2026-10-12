# Training and output code from the assessment notebook.
from pathlib import Path
import os, shutil
import json, hashlib, platform, time, zipfile, subprocess, sys, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
try:
    from IPython.display import display, Markdown, FileLink
except ImportError:
    def display(value): print(value)
    def FileLink(value): return str(value)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
SEED = 42
WORKSPACE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd().resolve()
ROOT = Path('/kaggle/working') if Path('/kaggle/working').exists() else WORKSPACE
OUT = ROOT / 'freight_submission'
OUT.mkdir(exist_ok=True, parents=True)
CONFIGS = [dict(max_leaf_nodes=15, l2_regularization=10.),
           dict(max_leaf_nodes=31, l2_regularization=10.),
           dict(max_leaf_nodes=31, l2_regularization=50.)]
MAX_ITER = 300
COLORS = ['#0f766e', '#2563eb', '#f59e0b', '#a855f7']
plt.rcParams.update({'figure.figsize':(10,4.5), 'figure.dpi':120,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','axes.labelcolor':'#334155',
                     'font.size':10,'axes.prop_cycle':matplotlib.cycler(color=COLORS)})
VERSIONS = {p:importlib.metadata.version(p) for p in ['numpy','pandas','matplotlib','scikit-learn','joblib']}
print('Python:', platform.python_version(), '| Versions:', VERSIONS)

FILE_ALIASES = {
    'train': ('train-test.csv', 'train_test.csv'),
    'validation': ('validation.csv',),
    'template': ('validation-predictions-template.csv', 'validation_predictions_template.csv'),
    'december': ('december-chart-inputs.csv', 'december_chart_inputs.csv'),
}

def paths_in(folder):
    return {key: next((folder / name for name in names if (folder / name).is_file()),
                      folder / names[0]) for key, names in FILE_ALIASES.items()}

# An explicit data folder takes priority over automatic discovery.
override = os.environ.get('SPOTTER_DATA_DIR')
if override:
    candidates = [Path(override).expanduser().resolve()]
else:
    candidates = []
    for base in dict.fromkeys([WORKSPACE, Path.cwd().resolve()]):
        for parent in [base, *base.parents]:
            candidates.extend([parent, parent / 'data', parent / 'upload'])
    kaggle_input = Path('/kaggle/input')
    if kaggle_input.is_dir():
        candidates.extend(sorted({p.parent for name in FILE_ALIASES['train']
                                  for p in kaggle_input.rglob(name)}))
    candidates = list(dict.fromkeys(candidates))
DATA_DIR = next((folder for folder in candidates
                 if all(p.is_file() for p in paths_in(folder).values())), None)
if DATA_DIR is None:
    checked = '\n'.join(str(folder) for folder in candidates)
    raise FileNotFoundError('Could not find all four assessment CSVs. Put them together in '
                            'the project folder, data/, or upload/, or set SPOTTER_DATA_DIR. '
                            'Checked:\n' + checked)
PATHS = paths_in(DATA_DIR)
print('Data directory:', DATA_DIR)
train = pd.read_csv(PATHS['train'])
validation = pd.read_csv(PATHS['validation'])
template = pd.read_csv(PATHS['template'])
december = pd.read_csv(PATHS['december'])
TARGET = 'posted_rate'
CORE = ['pickup','delivery','distance','equipment','weight','date']
FULL = CORE + ['pickup_lat','pickup_lon','delivery_lat','delivery_lon','market_index','quote_signal']
for label,frame in [('train',train),('validation',validation)]:
    required = ['load_id'] + FULL + ([TARGET] if label == 'train' else [])
    assert set(required).issubset(frame.columns), f'{label} missing required columns'
    assert frame.load_id.notna().all() and frame.load_id.is_unique
    frame['date'] = pd.to_datetime(frame['date'], errors='raise')
assert len(validation) == len(template) == 12000
assert list(template.columns) == ['load_id','predicted_rate']
assert set(template.load_id) == set(validation.load_id)
assert template.load_id.is_unique
assert set(train.load_id).isdisjoint(validation.load_id)
assert list(december.columns) == CORE + ['predicted_rate']
assert np.isfinite(train[TARGET]).all() and train[TARGET].gt(0).all()
assert train.date.max() < validation.date.min(), 'Expected forward prediction dates'
manifest = {k:{'filename':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in PATHS.items()}
display(pd.DataFrame([{'dataset':k,'rows':len(f),'columns':len(f.columns)} for k,f in
                      [('train',train),('validation',validation),('template',template),('december',december)]]))
print('Training:', train.date.min().date(), 'to', train.date.max().date())
print('Prediction:', validation.date.min().date(), 'to', validation.date.max().date())


quality = pd.DataFrame({'missing':train.isna().sum(),'dtype':train.dtypes.astype(str)})
display(quality)
print('Duplicate predictor rows:', train.duplicated(FULL).sum())
print('Unseen submission lanes:', len(set(zip(validation.pickup,validation.delivery)) - set(zip(train.pickup,train.delivery))))
monthly = train.groupby(train.date.dt.to_period('M')).agg(rows=(TARGET,'size'),median_rate=(TARGET,'median'))
fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].hist(train[TARGET],bins=55,color=COLORS[0],alpha=.85)
axes[0].set(title='Observed training load rates',xlabel='Posted rate ($)',ylabel='Loads')
axes[1].plot(monthly.index.astype(str),monthly.median_rate,marker='o',color=COLORS[1])
axes[1].set(title='Monthly median: changing load mix',xlabel='Training month',ylabel='Median posted rate ($)')
axes[1].tick_params(axis='x',rotation=45)
fig.tight_layout(); fig.savefig(OUT/'data_overview.png',bbox_inches='tight'); plt.show()
print('Monthly medians also depend on which loads appear that month.')


def features(frame, mode):
    """Build predictors from the fields available for this input set."""
    cols = FULL if mode == 'full' else CORE
    x = frame[cols].copy()
    d = pd.to_datetime(x.pop('date'), errors='raise')
    x['day_index'] = (d - pd.Timestamp('2025-01-01')).dt.days
    x['month'] = d.dt.month
    x['weekday'] = d.dt.dayofweek
    x['day_of_month'] = d.dt.day
    x['weekend'] = (d.dt.dayofweek >= 5).astype(int)
    x['annual_sin'] = np.sin(2*np.pi*d.dt.dayofyear/365.25)
    x['annual_cos'] = np.cos(2*np.pi*d.dt.dayofyear/365.25)
    x['lane'] = x['pickup'].astype(str) + ' → ' + x['delivery'].astype(str)
    for c in x.columns:
        if c not in ['pickup','delivery','equipment','lane']:
            x[c] = pd.to_numeric(x[c],errors='raise').replace([np.inf,-np.inf],np.nan)
    x['log_distance'] = np.log1p(x['distance'].clip(lower=0))
    x['weight_per_mile'] = x['weight'] / x['distance'].clip(lower=1)
    if mode == 'full':
        x['quote_times_distance'] = x['quote_signal'] * x['distance']
        x['market_times_distance'] = x['market_index'] * x['distance']
        x['latitude_delta'] = x['delivery_lat'] - x['pickup_lat']
        x['longitude_delta'] = x['delivery_lon'] - x['pickup_lon']
    return x
X = {mode:features(train,mode) for mode in ['full','core']}
# Keep the source units: miles, pounds and dollars.
def build_model(x, name, config=None):
    """Keep fitted preprocessing and regression in one pipeline."""
    cat = x.select_dtypes(include=['object','string']).columns.tolist()
    num = [c for c in x.columns if c not in cat]
    numeric = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    if name == 'ridge':
        categorical = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                                ('encode',OneHotEncoder(handle_unknown='ignore'))])
        estimator = Ridge(alpha=10.,solver='lsqr')
    else:
        categorical = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                                ('encode',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))])
        # Disable HGB's random internal holdout; validation is chronological.
        estimator = (DummyRegressor(strategy='median') if name == 'median' else
                     HistGradientBoostingRegressor(max_iter=MAX_ITER,learning_rate=.06,
                     early_stopping=False,random_state=SEED,**config))
    return Pipeline([('preprocess',ColumnTransformer([('numeric',numeric,num),('categorical',categorical,cat)])),
                     ('model',estimator)])
def predict_positive(model, x):
    """Keep rates positive to match the scorer requirements."""
    return np.maximum(model.predict(x), .01)
def metrics(y,p):
    """Measure dollar errors, R2 and mean absolute percentage error."""
    return {'MAE':float(mean_absolute_error(y,p)), 'RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'R2':float(r2_score(y,p)), 'MAPE_pct':float(np.mean(np.abs((np.asarray(y)-p)/np.asarray(y)))*100)}
print('Feature counts:', {k:v.shape[1] for k,v in X.items()})


FOLDS = [('August',pd.Timestamp('2025-08-01'),pd.Timestamp('2025-09-01')),
         ('September',pd.Timestamp('2025-09-01'),pd.Timestamp('2025-10-01'))]
# Keep October outside the search so it can test the selected settings.
TEST_START = pd.Timestamp('2025-10-01')
rows=[]
for label,start,end in FOLDS:
    fit = train.date < start
    dev = train.date.ge(start) & train.date.lt(end)
    assert fit.any() and dev.any() and train.loc[fit,'date'].max() < train.loc[dev,'date'].min()
    rows.append({'fold':label,'fit_rows':int(fit.sum()),'evaluate_rows':int(dev.sum()),
                 'fit_end':str(train.loc[fit,'date'].max().date()),'evaluate_start':str(start.date())})
split_table = pd.DataFrame(rows)
display(split_table)
print('Locked October rows:', int(train.date.ge(TEST_START).sum()))


records=[]; specifications={}; winners={}
for mode,x in X.items():
    specs = [('median',None),('ridge',None)] + [(f'hgb_{i+1}',cfg) for i,cfg in enumerate(CONFIGS)]
    for name,cfg in specs:
        specifications[(mode,name)] = cfg
        for fold,start,end in FOLDS:
            fit=train.date.lt(start); dev=train.date.ge(start)&train.date.lt(end)
            model=build_model(x,'hgb' if name.startswith('hgb') else name,cfg)
            began=time.perf_counter(); model.fit(x.loc[fit],train.loc[fit,TARGET])
            p=predict_positive(model,x.loc[dev])
            records.append({'mode':mode,'model':name,'fold':fold,'seconds':time.perf_counter()-began,
                            **metrics(train.loc[dev,TARGET],p)})
            print(mode,name,fold,'MAE:',round(records[-1]['MAE'],2))
    # Rank on development folds only; October does not influence this choice.
    ranks=pd.DataFrame(records).query('mode == @mode').groupby('model')['MAE'].mean().sort_values()
    winners[mode]=ranks.index[0]
results=pd.DataFrame(records)
results.to_csv(OUT/'development_metrics.csv',index=False)
leaderboard=results.groupby(['mode','model'])[['MAE','RMSE','MAPE_pct']].mean().reset_index()
display(leaderboard.sort_values(['mode','MAE']).round(3))
print('Development-selected models:',winners)
# Plot the development scores with the selected model highlighted.
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), sharex=True)
names = {'median': 'Median baseline', 'ridge': 'Ridge',
         'hgb_1': 'HGB 1', 'hgb_2': 'HGB 2', 'hgb_3': 'HGB 3'}
maximum = leaderboard.MAE.max() * 1.22
for ax, mode in zip(axes, ['full', 'core']):
    tab = leaderboard.query('mode == @mode').sort_values('MAE')
    selected_color = COLORS[0] if mode == 'full' else COLORS[1]
    colors = [selected_color if name == winners[mode] else '#b8c6d1'
              for name in tab.model]
    labels = [names[name] + (' (selected)' if name == winners[mode] else '')
              for name in tab.model]
    bars = ax.barh(labels, tab.MAE, color=colors, height=.62)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f'{value:,.1f}' for value in tab.MAE], padding=6, fontsize=10)
    ax.set(title='Full inputs' if mode == 'full' else 'Six December fields',
           xlabel='Mean August/September MAE ($)', xlim=(0, maximum))
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e2e8f0', linewidth=.7)
    ax.tick_params(axis='y', length=0, labelsize=10)
fig.tight_layout(pad=1.4, w_pad=3)
fig.savefig(OUT / 'model_comparison.png', dpi=180, bbox_inches='tight')
plt.show()


def selected(mode):
    """Build the settings chosen on the August and September folds."""
    name=winners[mode]
    return build_model(X[mode],'hgb' if name.startswith('hgb') else name,specifications[(mode,name)])
fit=train.date.lt(TEST_START); test=~fit
holdout={}; test_records=[]
rng=np.random.default_rng(SEED)
for mode in ['full','core']:
    m=selected(mode); m.fit(X[mode].loc[fit],train.loc[fit,TARGET])
    p=predict_positive(m,X[mode].loc[test])
    holdout[mode]=pd.DataFrame({'date':train.loc[test,'date'].to_numpy(),
                               'actual':train.loc[test,TARGET].to_numpy(),'predicted':p})
    # Resample dates rather than rows to keep same-day loads together.
    daily=holdout[mode].assign(error=lambda f:abs(f.actual-f.predicted)).groupby('date').error.agg(['sum','count'])
    idx=rng.integers(0,len(daily),size=(1000,len(daily)))
    bootstrap=daily['sum'].to_numpy()[idx].sum(axis=1)/daily['count'].to_numpy()[idx].sum(axis=1)
    lo,hi=np.quantile(bootstrap,[.025,.975])
    test_records.append({'mode':mode,'model':winners[mode],**metrics(holdout[mode].actual,p),
                         'MAE_day_bootstrap_low':float(lo),'MAE_day_bootstrap_high':float(hi)})
test_metrics=pd.DataFrame(test_records)
test_metrics.to_csv(OUT/'october_test_metrics.csv',index=False)
display(test_metrics.round(3))
# Use the same limits and units on both axes to show prediction errors fairly.
from matplotlib.ticker import StrMethodFormatter
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.7))
axis_limit = max(max(h.actual.max(), h.predicted.max()) for h in holdout.values()) * 1.05
for ax, mode in zip(axes, ['full', 'core']):
    h = holdout[mode]
    sample = h.sample(min(1500, len(h)), random_state=SEED)
    color = COLORS[0] if mode == 'full' else COLORS[1]
    ax.scatter(sample.actual, sample.predicted, s=12, alpha=.45,
               color=color, edgecolors='none', rasterized=True)
    ax.plot([0, axis_limit], [0, axis_limit], color='#dc2626',
            linestyle='--', linewidth=1.3, label='Predicted = actual')
    ax.set(title='Full inputs' if mode == 'full' else 'Six December fields',
           xlabel='Actual posted rate ($)', ylabel='Predicted rate ($)',
           xlim=(0, axis_limit), ylim=(0, axis_limit))
    ax.set_aspect('equal', adjustable='box')
    ax.set_axisbelow(True)
    ax.grid(color='#e2e8f0', linewidth=.6)
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    row = test_metrics.set_index('mode').loc[mode]
    ax.text(.035, .965, f'MAE ${row.MAE:,.2f}\nRMSE ${row.RMSE:,.2f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox={'facecolor': 'white', 'edgecolor': '#e2e8f0', 'pad': 6})
    ax.legend(loc='lower right', fontsize=9, frameon=False)
fig.tight_layout(pad=1.2, w_pad=2.5)
fig.savefig(OUT / 'october_diagnostics.png', dpi=180, bbox_inches='tight')
plt.show()
segment=holdout['full'].copy(); segment['equipment']=train.loc[test,'equipment'].to_numpy()
display(segment.assign(abs_error=lambda f:abs(f.actual-f.predicted)).groupby('equipment').agg(
    loads=('abs_error','size'),MAE=('abs_error','mean')).round(2))


# This horizon check overlaps development data; it is not a second independent test.
horizon_records=[]
for mode in ['full','core']:
    m=selected(mode); mask=train.date.lt('2025-09-01')
    m.fit(X[mode].loc[mask],train.loc[mask,TARGET])
    for month in ['2025-09','2025-10']:
        mask_eval=train.date.dt.strftime('%Y-%m').eq(month)
        p=predict_positive(m,X[mode].loc[mask_eval])
        horizon_records.append({'mode':mode,'month':month,'horizon_months':1 if month=='2025-09' else 2,
                                **metrics(train.loc[mask_eval,TARGET],p)})
horizon_metrics=pd.DataFrame(horizon_records)
horizon_metrics.to_csv(OUT/'horizon_diagnostics.csv',index=False)
display(horizon_metrics.round(3))


final_models={}
for mode in ['full','core']:
    final_models[mode]=selected(mode).fit(X[mode],train[TARGET])
    joblib.dump(final_models[mode],OUT/f'{mode}_model.joblib')
raw_predictions=pd.DataFrame({'load_id':validation.load_id,
                              'predicted_rate':predict_positive(final_models['full'],features(validation,'full'))})
# Match by ID and retain template order instead of assuming CSV row alignment.
submission=template[['load_id']].merge(raw_predictions,on='load_id',how='left',validate='one_to_one',sort=False)
assert submission.load_id.tolist()==template.load_id.tolist()
assert submission.predicted_rate.notna().all()
submission.to_csv(OUT/'validation_predictions.csv',index=False)
# The core model uses exactly the fields supplied for December.
december_completed=december.copy()
december_completed['predicted_rate']=predict_positive(final_models['core'],features(december,'core'))
(OUT/'data').mkdir(exist_ok=True)
december_completed.to_csv(OUT/'data'/'december_chart_inputs.csv',index=False)
display(submission.head()); display(december_completed.head())
print('Saved validation and December predictions.')


# Use the supplied scorer without changing its checks or chart.
scorer_dir=OUT/'scorer_results'
scorer_path=OUT/'score.py'
scorer_path.write_text('from __future__ import annotations\n\nimport argparse\nfrom pathlib import Path\n\nimport matplotlib\n\nmatplotlib.use("Agg")\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n\n\nEXPECTED_ROWS = 12_000\nEXPECTED_IDS = {f"TE-{index:06d}" for index in range(1, EXPECTED_ROWS + 1)}\nDECEMBER_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")\nFIXED_PICKUP = "Lexington"\nFIXED_DELIVERY = "Fort Wayne"\nFIXED_DISTANCE = 360.0\nFIXED_EQUIPMENT = "Dry Van"\nFIXED_WEIGHT = 32_000.0\n\n\ndef fail(message: str) -> None:\n    raise SystemExit(f"ERROR: {message}")\n\n\ndef read_csv(path: Path, label: str) -> pd.DataFrame:\n    if not path.is_file():\n        fail(f"{label} file not found: {path}")\n    try:\n        return pd.read_csv(path)\n    except Exception as exc:\n        fail(f"could not read {label}: {exc}")\n\n\ndef numeric_series(frame: pd.DataFrame, column: str, label: str) -> pd.Series:\n    values = pd.to_numeric(frame[column], errors="coerce")\n    if values.isna().any() or not np.isfinite(values).all():\n        fail(f"{label} contains invalid {column} values")\n    return values.astype(float)\n\n\ndef validate_predictions(predictions: pd.DataFrame) -> None:\n    if list(predictions.columns) != ["load_id", "predicted_rate"]:\n        fail("predictions must contain exactly two columns in this order: load_id,predicted_rate")\n    if len(predictions) != EXPECTED_ROWS:\n        fail(f"predictions must contain exactly {EXPECTED_ROWS:,} rows")\n    if predictions["load_id"].isna().any() or predictions["load_id"].duplicated().any():\n        fail("predictions contains missing or duplicate load_id values")\n\n    submitted_ids = set(predictions["load_id"].astype(str))\n    missing = EXPECTED_IDS - submitted_ids\n    extra = submitted_ids - EXPECTED_IDS\n    if missing or extra:\n        fail(\n            "prediction IDs do not match the validation set "\n            f"(missing={len(missing)}, extra={len(extra)})"\n        )\n\n    predicted_rate = numeric_series(predictions, "predicted_rate", "predictions")\n    if (predicted_rate <= 0).any():\n        fail("predictions contains non-positive predicted_rate values")\n\n\ndef validate_december(frame: pd.DataFrame) -> pd.DataFrame:\n    columns = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]\n    if list(frame.columns) != columns:\n        fail("December predictions must keep the original seven columns and column order")\n\n    result = frame.copy()\n    result["date"] = pd.to_datetime(result["date"], errors="coerce")\n    if result["date"].isna().any():\n        fail("December predictions contains invalid dates")\n    result["distance"] = numeric_series(result, "distance", "December predictions")\n    result["weight"] = numeric_series(result, "weight", "December predictions")\n    result["predicted_rate"] = numeric_series(result, "predicted_rate", "December predictions")\n\n    if result["date"].duplicated().any():\n        fail("December predictions contains duplicate dates")\n    if len(result) != 31 or set(result["date"]) != set(DECEMBER_DATES):\n        fail("December predictions must contain one row for every day from 2025-12-01 to 2025-12-31")\n    if not result["pickup"].eq(FIXED_PICKUP).all():\n        fail(f"December pickup must be {FIXED_PICKUP} for all rows")\n    if not result["delivery"].eq(FIXED_DELIVERY).all():\n        fail(f"December delivery must be {FIXED_DELIVERY} for all rows")\n    if not np.isclose(result["distance"], FIXED_DISTANCE).all():\n        fail(f"December distance must be {FIXED_DISTANCE:g} for all rows")\n    if not result["equipment"].eq(FIXED_EQUIPMENT).all():\n        fail(f"December equipment must be {FIXED_EQUIPMENT} for all rows")\n    if not np.isclose(result["weight"], FIXED_WEIGHT).all():\n        fail(f"December weight must be {FIXED_WEIGHT:g} for all rows")\n    if (result["predicted_rate"] <= 0).any():\n        fail("December predicted_rate values must be positive")\n    return result.sort_values("date")\n\n\ndef save_december_chart(december: pd.DataFrame, output: Path) -> None:\n    figure, axis = plt.subplots(figsize=(10.8, 4.8), dpi=180)\n    color = "#064A56"\n    axis.plot(\n        december["date"],\n        december["predicted_rate"],\n        color=color,\n        linewidth=2.6,\n        marker="o",\n        markersize=3.2,\n    )\n    floor = float(december["predicted_rate"].min())\n    axis.fill_between(\n        december["date"],\n        december["predicted_rate"],\n        floor - max(10.0, floor * 0.02),\n        color=color,\n        alpha=0.08,\n    )\n    axis.set_title("Candidate: December 2025 Predicted Load Rate", loc="left", fontsize=15, fontweight="bold", pad=12)\n    axis.set_ylabel("Predicted rate ($)")\n    axis.grid(axis="y", color="#D9E2E4", linewidth=0.8)\n    axis.spines[["top", "right"]].set_visible(False)\n    axis.spines[["left", "bottom"]].set_color("#9DAFB3")\n    axis.tick_params(axis="x", rotation=35)\n    axis.text(\n        0,\n        -0.40,\n        "Fixed inputs: Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb | only date changes",\n        transform=axis.transAxes,\n        fontsize=9.5,\n        color="#455A60",\n    )\n    figure.tight_layout(rect=(0, 0.12, 1, 1))\n    figure.savefig(output, bbox_inches="tight")\n    plt.close(figure)\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser(\n        description="Validate candidate output files and generate the fixed December chart."\n    )\n    parser.add_argument("--predictions", required=True, help="CSV with load_id,predicted_rate")\n    parser.add_argument(\n        "--december-predictions",\n        required=True,\n        help="Completed data/december_chart_inputs.csv",\n    )\n    parser.add_argument("--output-dir", default="scorer_results")\n    args = parser.parse_args()\n\n    validate_predictions(read_csv(Path(args.predictions), "predictions"))\n    december = validate_december(read_csv(Path(args.december_predictions), "December predictions"))\n    output = Path(args.output_dir)\n    output.mkdir(parents=True, exist_ok=True)\n    chart = output / "candidate_december.png"\n    save_december_chart(december, chart)\n\n    print(f"Validated {EXPECTED_ROWS:,} final predictions.")\n    print("Validated 31 fixed December predictions.")\n    print(f"Created chart: {chart}")\n    print("Final validation metrics are calculated by Spotter after submission.")\n\n\nif __name__ == "__main__":\n    main()\n', encoding='utf-8')
completed=subprocess.run([sys.executable,str(scorer_path),
    '--predictions',str(OUT/'validation_predictions.csv'),
    '--december-predictions',str(OUT/'data'/'december_chart_inputs.csv'),
    '--output-dir',str(scorer_dir)],capture_output=True,text=True,check=True)
print(completed.stdout)
fig,ax=plt.subplots(figsize=(11,5)); ax.imshow(plt.imread(scorer_dir/'candidate_december.png'))
ax.axis('off'); plt.tight_layout(); plt.show()


import textwrap

report_path = OUT / 'Freight_Rate_Validation_Report.pdf'

def report_footer(fig, number):
    fig.text(.06, .035, 'Arafat Rahman | Spotter ML Engineer assessment',
             fontsize=8.5, color='#64748b')
    fig.text(.94, .035, f'{number} / 4', ha='right', fontsize=8.5, color='#64748b')

def report_section(fig, heading, body, y):
    lines = textwrap.fill(body, width=91)
    fig.text(.07, y, heading, fontsize=11.5, weight='bold', color='#123047', va='top')
    fig.text(.07, y-.029, lines, fontsize=10, color='#334155', va='top', linespacing=1.3, parse_math=False)
    return y - .029 - .017 * len(lines.splitlines()) - .022

with PdfPages(report_path) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    fig.text(.07, .945, 'Freight rate prediction', fontsize=24, weight='bold', color='#123047')
    fig.text(.07, .908, 'Validation approach and results', fontsize=12, color='#0f766e')
    y = .86
    new_routes = len(set(zip(validation.pickup, validation.delivery)) - set(zip(train.pickup, train.delivery)))
    y = report_section(fig, 'Data and checks',
        f'{len(train):,} labeled loads cover January-October 2025. There are {len(validation):,} '
        f'unlabeled November/December loads. Training has {train.weight.isna().sum():,} missing weights '
        f'and {train.market_index.isna().sum():,} missing market index values; numeric fields use median '
        f'imputation. There are {train.duplicated(FULL).sum():,} duplicate predictor rows and '
        f'{new_routes:,} pickup/delivery pairs absent from training.', y)
    y = report_section(fig, 'Training and evaluation',
        'Train through July to evaluate August, then through August to evaluate September. '
        'Select settings by mean development MAE and reserve October for testing. '
        'Imputers and encoders fit within each training split; IDs are excluded. '
        'The final models refit on all labeled rows.', y)
    y = report_section(fig, 'Model choice',
        f'Compare a median baseline, Ridge and three histogram gradient boosting settings. '
        f'The selected full model is {winners["full"]}; the six-field model is {winners["core"]}. '
        'December uses a separate model because its inputs lack coordinates and market signals.', y)

    fig.text(.07, y, 'October holdout', fontsize=11.5, weight='bold', color='#123047', va='top')
    rows = []
    for mode in ['full', 'core']:
        r = test_metrics.set_index('mode').loc[mode]
        rows.append(['Full' if mode == 'full' else 'Six fields', r.model,
                     f'${r.MAE:,.2f}', f'${r.RMSE:,.2f}', f'{r.R2:.3f}', f'{r.MAPE_pct:.2f}%'])
    table_ax = fig.add_axes([.07, y-.138, .86, .103])
    table_ax.axis('off')
    table = table_ax.table(cellText=rows, colLabels=['Inputs', 'Model', 'MAE', 'RMSE', 'R2', 'MAPE'],
                          cellLoc='center', colWidths=[.19, .16, .17, .17, .13, .18], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#e2e8f0')
        cell.set_linewidth(.7)
        cell.set_facecolor('#123047' if row == 0 else ('#f1f5f9' if row % 2 else 'white'))
        cell.get_text().set_color('white' if row == 0 else '#334155')
        if row == 0: cell.get_text().set_weight('bold')
    y -= .165
    intervals = test_metrics.set_index('mode')
    y = report_section(fig, 'MAE intervals',
        f'Resampling October days 1,000 times gives 95% intervals of '
        f'${intervals.loc["full", "MAE_day_bootstrap_low"]:.2f}-${intervals.loc["full", "MAE_day_bootstrap_high"]:.2f} '
        f'for full inputs and ${intervals.loc["core", "MAE_day_bootstrap_low"]:.2f}-'
        f'${intervals.loc["core", "MAE_day_bootstrap_high"]:.2f} for six fields. '
        'These describe variation across October days, not uncertainty for each future load.', y)
    y = report_section(fig, 'Limits',
        'Future labels are unavailable. The two-month check overlaps September development data. '
        'Ten training months only partly cover annual seasonality, and trees have limited trend '
        'extrapolation. quote_signal is supplied for prediction; its availability before the '
        'quoted rate needs checking in production. The scorer checks file validity, not hidden-label accuracy.', y)
    assert y > .055, 'Summary content needs another page or shorter text'
    report_footer(fig, 1)
    pdf.savefig(fig)
    plt.close(fig)

    december_rates = pd.read_csv(OUT / 'data' / 'december_chart_inputs.csv').predicted_rate
    spread = december_rates.max() - december_rates.min()
    pages = [
        ('Model comparison', 'August and September development folds | lower MAE is better',
         OUT / 'model_comparison.png',
         'Each label shows mean development MAE in dollars. Colored bars mark the selected settings. '
         'Full-input HGB 1 and HGB 3 scores are close; October is excluded from this comparison.'),
        ('October prediction errors', f'{len(holdout["full"]):,} held-out loads | '
         f'{min(1500, len(holdout["full"])):,} loads shown per model',
         OUT / 'october_diagnostics.png',
         'Both axes use the same scale. The dashed line marks an exact prediction. '
         'Some high-rate loads are underpredicted, which helps explain why RMSE exceeds MAE.'),
        ('December fixed-route forecast', 'Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb',
         scorer_dir / 'candidate_december.png',
         f'The original scorer chart uses a zoomed y-axis. Rates span '
         f'${december_rates.min():.2f}-${december_rates.max():.2f}: a ${spread:.2f} range '
         f'({100*spread/december_rates.mean():.1f}% of the mean). '
         'Only date changes. These are model forecasts; future labels are unavailable.')
    ]
    for number, (title, subtitle, image_path, note) in enumerate(pages, start=2):
        fig = plt.figure(figsize=(11.69, 7.4), facecolor='white')
        fig.text(.06, .93, title, fontsize=22, weight='bold', color='#123047')
        fig.text(.06, .88, subtitle, fontsize=10.5, color='#475569')
        image_ax = fig.add_axes([.045, .18, .91, .65])
        image_ax.imshow(plt.imread(image_path))
        image_ax.axis('off')
        fig.text(.06, .12, textwrap.fill(note, width=137), fontsize=9.5,
                 color='#475569', va='top', linespacing=1.3, parse_math=False)
        report_footer(fig, number)
        pdf.savefig(fig)
        plt.close(fig)
print('Report:', report_path)


metadata={'seed':SEED,'versions':VERSIONS,'input_files':manifest,'selected_models':winners,
          'target':TARGET,'max_iter':MAX_ITER,'configs':CONFIGS,
          'development_folds':split_table.to_dict('records'),'october_metrics':test_records,
          'limitations':['No hidden validation labels','Core model required for December chart',
                         'Ten months of training only partly cover annual seasonality','Market signal availability assumed']}
(OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2), encoding='utf-8')
(OUT/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in VERSIONS.items())+'\n', encoding='utf-8')
(OUT/'README.md').write_text('# Freight rate prediction\n\nThese files come from the Spotter assessment pipeline.\n\nTo repeat the run, install requirements.txt, put the four original CSVs beside\nfreight_pipeline.py (or in data/ or upload/), and run `python freight_pipeline.py`.\nSet SPOTTER_DATA_DIR for a different input folder. For the notebook, run the\ncells in order. On Kaggle, attach the dataset first. CPU is enough.\n\nAugust and September select the model settings. October is the held-out test.\nThe final fit uses all labeled rows. Validation loads use the full input set;\nthe December chart uses the six available fields.\n\nvalidation_predictions.csv contains the 12,000 final predictions. The completed\nDecember inputs are in data/december_chart_inputs.csv. The original score.py checks\nboth files and creates scorer_results/candidate_december.png. The PDF explains\nthe splits and results, and includes that chart. Run details and metrics are\nsaved alongside it. Hidden validation labels are not available.\n\nThe original datasets are not included. The Loom recording is a separate item.\n', encoding='utf-8')
with (OUT/'requirements.txt').open('a', encoding='utf-8') as f: f.write('ipython>=8,<10\n')
(OUT/'Loom_outline.md').write_text("# Loom walkthrough notes\n\nAim for 2-3 minutes. Keep the notebook and completed outputs open.\n\n0:00-0:30 - Explain posted_rate and the January-October training period. Show\nthe missing weights and market index values, plus the new routes in validation.\n\n0:30-1:10 - Show the date splits. August and September choose the settings;\nOctober tests the selected models. Explain why random splitting would be a\nweaker match for predicting later months. Point out that imputation and encoding\nare fitted within each split.\n\n1:10-1:45 - Show the baseline comparison and October metrics. Explain the\nlowest-development-MAE choice and discuss the large errors visible in the plot.\nThe full model's October MAE is FULL_MAE; the core model's is CORE_MAE.\n\n1:45-2:15 - Show features() and the two input sets. The December file lacks\nmarket signals and coordinates, which is why it uses a separate six-field model.\nMention the short seasonal history and the unavailable future labels.\n\n2:15-2:45 - Show the ID-matched CSV, the scorer output and the December chart\nin the report. Finish with the README command for repeating the run.\n".replace('FULL_MAE', f'${test_records[0]["MAE"]:.2f}').replace('CORE_MAE', f'${test_records[1]["MAE"]:.2f}'), encoding='utf-8')


# Export current code; skip this packaging cell in the script.
source_candidates = [WORKSPACE / 'spotter-freight-rate-prediction-ml-assessment.ipynb',
                     DATA_DIR / 'spotter-freight-rate-prediction-ml-assessment.ipynb',
                     Path('/kaggle/working/__notebook__.ipynb'),
                     ROOT / 'Spotter_Freight_Rate_Kaggle.ipynb']
notebook_source = next((p for p in source_candidates if p.is_file()), None)
export_path = OUT / 'freight_pipeline.py'
if notebook_source:
    nb = json.loads(notebook_source.read_text(encoding='utf-8'))
    scripts = [''.join(cell['source']) for cell in nb['cells']
               if cell['cell_type'] == 'code'
               and '# Export current code;' not in ''.join(cell['source'])]
    export_path.write_text('\n\n'.join(scripts), encoding='utf-8')
elif '__file__' in globals():
    if Path(__file__).resolve() != export_path.resolve():
        shutil.copy2(__file__, export_path)
else:
    print('Notebook source was not found. Include the notebook separately in the handoff.')
archive = ROOT / 'Spotter_Freight_Submission.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.suffix != '.joblib': z.write(p, p.relative_to(OUT))
    if notebook_source: z.write(notebook_source, notebook_source.name)
print('Generated files:')
for p in sorted(OUT.rglob('*')):
    if p.is_file(): print(p.relative_to(OUT))
display(FileLink(str(archive)))
display(FileLink(str(OUT / 'validation_predictions.csv')))
display(FileLink(str(report_path)))
