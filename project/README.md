# Damage Characterisation of GFRP Composites under Impact and Compression using Acoustic Emission and Data-Driven Analysis

Unsupervised clustering of 30,574 AE hits (8 GFRP/cenosphere specimens, CAI and pure compression) into
damage-mechanism groups, physical validation with RA–AF and loading history, and a Random Forest model
for instant classification of new AE files.

## Run (VS Code terminal)
```
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python ae_clustering_analysis.py
```
Outputs: `figures/` (charts), `results/` (CSV tables), `models/ae_cluster_model.joblib`.

## Demo app
```
streamlit run app.py
```
Modules: Home, Damage Clustering, Model Validation, Source Study Comparison, Predict New AE Data, Train on Your Own Data.

- Prediction demo: upload `data/sample_new_ae_file.csv` on the Predict page.
- Any-material demo: upload `data/demo_single_test.csv` on the Train on Your Own Data page, click Run clustering, then "Use this model on the Predict page".

The built-in model is specific to the GFRP/cenosphere laminates of this study. The Train on Your Own Data page repeats the same workflow on any AE file that has amplitude, energy, duration, counts and rise time columns.

## Method
1. Features: Amplitude, Energy, Duration, Counts, Risetime + RA (rise time / amplitude) and AF (counts / duration); log-scaled and standardised.
2. K-means; k = 4 chosen by Davies–Bouldin index; stability checked with random seeds and bootstrap samples; compared with a Gaussian mixture model.
3. Clusters named from centroid physics (RA–AF → tensile vs shear, energy → low vs high).
4. Validation: RA–AF plot, damage progression over loading time, CAI vs pure compression.
5. Random Forest trained on cluster labels, benchmarked against kNN (same features, same splits); evaluated with a 90/10 hold-out and leave-one-specimen-out (clustering refitted on training data only in each fold).
6. Prediction confidence: RF predictions below 70% probability are flagged as uncertain.
