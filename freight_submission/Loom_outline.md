# Loom walkthrough notes

Aim for 2-3 minutes. Keep the notebook and completed outputs open.

0:00-0:30 - Explain posted_rate and the January-October training period. Show
the missing weights and market index values, plus the new routes in validation.

0:30-1:10 - Show the date splits. August and September choose the settings;
October tests the selected models. Explain why random splitting would be a
weaker match for predicting later months. Point out that imputation and encoding
are fitted within each split.

1:10-1:45 - Show the baseline comparison and October metrics. Explain the
lowest-development-MAE choice and discuss the large errors visible in the plot.
The full model's October MAE is $196.66; the core model's is $169.05.

1:45-2:15 - Show features() and the two input sets. The December file lacks
market signals and coordinates, which is why it uses a separate six-field model.
Mention the short seasonal history and the unavailable future labels.

2:15-2:45 - Show the ID-matched CSV, the scorer output and the December chart
in the report. Finish with the README command for repeating the run.
