"""
Damage Characterisation of GFRP Composites under Impact and Compression
using Acoustic Emission and Data-Driven Analysis
------------------------------------------------------------------------
Unsupervised AE damage clustering + Random Forest deployment model.

Run:  python ae_clustering_analysis.py
Input : data/master_labeled.csv
Output: figures/*.png, results/*.csv, models/ae_cluster_model.joblib
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, adjusted_mutual_info_score,
                             accuracy_score, f1_score, classification_report,
                             confusion_matrix)

BASE = Path(__file__).parent
DATA = BASE / "data" / "master_labeled.csv"
FIG, RES, MOD = BASE / "figures", BASE / "results", BASE / "models"
for p in (FIG, RES, MOD):
    p.mkdir(exist_ok=True)

SEED = 42
K = 4                                   # chosen from Davies-Bouldin + stability (Step 2)
FEATURES = ["Amplitude", "Energy", "Duration", "Counts", "Risetime", "RA", "AF"]
LOG_FEATURES = ["Energy", "Duration", "Counts", "Risetime", "RA", "AF"]
FREQ_FEATURES = ["PeakFreq", "FreqCentroid"]
CONF_THRESHOLD = 0.70                   # RF predictions below this probability are flagged "uncertain"

# Colours (validated categorical palette) and cluster display order
ORDER = ["LE-Tensile", "HE-Tensile", "LE-Shear", "HE-Shear"]
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
LONG = {"LE-Tensile": "Low-energy tensile (matrix micro-cracking)",
        "HE-Tensile": "High-energy tensile (fibre-dominated)",
        "LE-Shear":   "Low-energy shear (interface / debonding)",
        "HE-Shear":   "High-energy shear (delamination growth)"}

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "font.size": 10,
                     "axes.axisbelow": True})


# ----------------------------------------------------------------------
# Step 1: load data and derive RA / AF
# ----------------------------------------------------------------------
def load_data(path=DATA):
    df = pd.read_csv(path)
    amp_v = 1e-6 * 10 ** (df["Amplitude"] / 20)          # dB(AE) -> volts
    df["RA"] = (df["Risetime"] / 1000) / amp_v            # ms/V
    df["AF"] = df["Counts"] / df["Duration"] * 1000       # kHz
    # normalised loading time per specimen (0 = first hit, 1 = last hit)
    t = df.groupby("specimen")["Time"]
    df["t_norm"] = (df["Time"] - t.transform("min")) / (t.transform("max") - t.transform("min"))
    df["stage"] = pd.cut(df["t_norm"], [-0.01, 1/3, 2/3, 1.01], labels=["Early", "Mid", "Late"])
    return df


def transform(df, cols=FEATURES):
    return pd.DataFrame({c: np.log1p(df[c]) if c in LOG_FEATURES else df[c] for c in cols})


def name_clusters(km, scaler, cols=FEATURES):
    """Name clusters from centroid physics: RA/AF -> tensile vs shear, Energy -> low/high."""
    cen = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=cols)
    shear = cen["RA"] - cen["AF"]                     # high RA & low AF -> shear
    names = {}
    tens = shear.nsmallest(2).index
    for grp, label in ((tens, "Tensile"), (shear.index.difference(tens), "Shear")):
        e = cen.loc[grp, "Energy"]
        names[e.idxmin()] = f"LE-{label}"
        names[e.idxmax()] = f"HE-{label}"
    return names


# ----------------------------------------------------------------------
# Step 2: choose number of clusters
# ----------------------------------------------------------------------
def select_k(X, ks=range(2, 9)):
    rows = []
    for k in ks:
        lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
        rows.append({"k": k,
                     "silhouette": silhouette_score(X, lab, sample_size=8000, random_state=0),
                     "davies_bouldin": davies_bouldin_score(X, lab)})
    res = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for a, col, lab, best in ((ax[0], "silhouette", "Silhouette (higher = better)", None),
                              (ax[1], "davies_bouldin", "Davies-Bouldin (lower = better)", K)):
        a.plot(res.k, res[col], color="#2a78d6", lw=2, marker="o", ms=6)
        a.set_xlabel("Number of clusters k"); a.set_title(lab, loc="left", fontsize=10)
    ax[1].scatter([K], res.loc[res.k == K, "davies_bouldin"], s=120, facecolors="none",
                  edgecolors="#0b0b0b", lw=1.5, zorder=5)
    ax[1].annotate(f"k = {K} selected", (K, res.loc[res.k == K, "davies_bouldin"].item()),
                   xytext=(-20, 25), textcoords="offset points")
    fig.tight_layout(); fig.savefig(FIG / "01_cluster_number_selection.png"); plt.close(fig)
    return res


# ----------------------------------------------------------------------
# Step 3: stability and method comparison
# ----------------------------------------------------------------------
def stability(X, base):
    seeds = [adjusted_rand_score(base, KMeans(K, n_init=10, random_state=s).fit_predict(X))
             for s in range(1, 6)]
    rng = np.random.default_rng(0); boot = []
    for i in range(5):
        idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        boot.append(adjusted_rand_score(base, KMeans(K, n_init=10, random_state=i).fit(X[idx]).predict(X)))
    g = GaussianMixture(K, random_state=SEED).fit(X).predict(X)
    return pd.DataFrame([
        {"check": "K-means, 5 random seeds (min ARI)", "value": min(seeds)},
        {"check": "K-means, 5 bootstrap 80% samples (mean ARI)", "value": np.mean(boot)},
        {"check": "K-means silhouette", "value": silhouette_score(X, base, sample_size=8000, random_state=0)},
        {"check": "K-means Davies-Bouldin", "value": davies_bouldin_score(X, base)},
        {"check": "GMM silhouette", "value": silhouette_score(X, g, sample_size=8000, random_state=0)},
        {"check": "GMM Davies-Bouldin", "value": davies_bouldin_score(X, g)},
    ])


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------
def stacked_share(tab, fname, title, xlabel):
    tab = tab[ORDER] * 100
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(tab) + 1.6))
    left = np.zeros(len(tab))
    for c in ORDER:
        ax.barh(tab.index.astype(str), tab[c], left=left, color=COLORS[c],
                edgecolor="white", linewidth=2, label=c, height=0.62)
        for y, (l, v) in enumerate(zip(left, tab[c])):
            if v >= 7:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="#0b0b0b")
        left += tab[c].values
    ax.set_xlim(0, 100); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left", fontsize=10)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
    ax.set_title(title, loc="left", fontsize=10, pad=24)
    fig.tight_layout(); fig.savefig(FIG / fname, bbox_inches="tight"); plt.close(fig)


def scatter_clusters(df, x, y, fname, title, xlabel, ylabel, logx=False, logy=False, n=6000):
    s = df.sample(min(n, len(df)), random_state=0)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for c in ORDER:
        p = s[s.cluster == c]
        ax.scatter(p[x], p[y], s=9, alpha=0.55, color=COLORS[c], edgecolors="none", label=c)
    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=10)
    ax.legend(markerscale=2.5, frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)


# ----------------------------------------------------------------------
# Step 6: Random Forest deployment model (leak-free evaluation)
# ----------------------------------------------------------------------
def rf():
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, class_weight="balanced")


def knn(k):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))


def tune_knn(X, y):
    gs = GridSearchCV(knn(5), {"kneighborsclassifier__n_neighbors": [3, 5, 11, 21, 41]}, cv=5, n_jobs=-1)
    gs.fit(X, y)
    return gs.best_params_["kneighborsclassifier__n_neighbors"]


def evaluate_fold(df, Z, tr, te, k_nn):
    """Scaler + K-means fitted on training rows only; each classifier must reproduce
    the cluster assignment of unseen hits from the same raw AE parameters."""
    sc = StandardScaler().fit(Z.iloc[tr])
    km = KMeans(K, n_init=10, random_state=SEED).fit(sc.transform(Z.iloc[tr]))
    y_tr, y_te = km.labels_, km.predict(sc.transform(Z.iloc[te]))
    out = {}
    for name, model in (("Random Forest", rf()), (f"kNN (k={k_nn})", knn(k_nn))):
        p = model.fit(df[FEATURES].iloc[tr], y_tr).predict(df[FEATURES].iloc[te])
        out[name] = (accuracy_score(y_te, p), f1_score(y_te, p, average="macro"))
    return out


def main():
    df = load_data()
    Z = transform(df)
    scaler = StandardScaler().fit(Z)
    X = scaler.transform(Z)
    print(f"Loaded {len(df):,} AE hits, {df.specimen.nunique()} specimens")

    # Step 2 - number of clusters
    ksel = select_k(X); ksel.to_csv(RES / "01_k_selection.csv", index=False)

    # Step 3 - final clustering
    km = KMeans(K, n_init=10, random_state=SEED).fit(X)
    names = name_clusters(km, scaler)
    df["cluster"] = pd.Series(km.labels_).map(names)
    stab = stability(X, km.labels_); stab.to_csv(RES / "02_stability_and_method_comparison.csv", index=False)
    print(stab.round(3).to_string(index=False))

    prof = df.groupby("cluster")[FEATURES + FREQ_FEATURES].median().loc[ORDER]
    prof.insert(0, "hits", df.cluster.value_counts()[ORDER])
    prof.insert(1, "share_%", (prof.hits / len(df) * 100).round(1))
    prof["interpretation"] = [LONG[c] for c in ORDER]
    prof.round(1).to_csv(RES / "03_cluster_profiles.csv")
    print(prof.round(1).to_string())

    # PCA view and RA-AF view
    pcs = PCA(2, random_state=SEED).fit(X); P = pcs.transform(X)
    df["PC1"], df["PC2"] = P[:, 0], P[:, 1]
    ev = pcs.explained_variance_ratio_ * 100
    scatter_clusters(df, "PC1", "PC2", "02_pca_clusters.png", "AE hit clusters (PCA projection)",
                     f"PC1 ({ev[0]:.0f}% variance)", f"PC2 ({ev[1]:.0f}% variance)")
    scatter_clusters(df, "RA", "AF", "03_ra_af_classification.png",
                     "RA-AF: tensile (low RA, high AF) vs shear (high RA, low AF)",
                     "RA value (ms/V, log scale)", "Average frequency AF (kHz, log scale)",
                     logx=True, logy=True)

    # Step 4 - damage progression over loading history
    stage = pd.crosstab(df.stage, df.cluster, normalize="index")
    stage.round(3).to_csv(RES / "04_cluster_share_by_loading_stage.csv")
    stacked_share(stage, "04_damage_progression.png",
                  "Damage-mechanism share across the loading history", "Share of AE hits (%)")

    test = pd.crosstab(df.test_type, df.cluster, normalize="index")
    test.round(3).to_csv(RES / "05_cluster_share_by_test_type.csv")
    stacked_share(test, "05_cai_vs_pure_compression.png",
                  "Mechanism share: impacted (CAI) vs non-impacted (pure compression)", "Share of AE hits (%)")

    spec = pd.crosstab(df.specimen, df.cluster, normalize="index")
    spec.round(3).to_csv(RES / "06_cluster_share_by_specimen.csv")
    stacked_share(spec, "06_cluster_share_by_specimen.png",
                  "Mechanism share by specimen (filler % / test type)", "Share of AE hits (%)")

    # Step 4b - comparison with the source study [1] (3 wt% cenosphere vs neat resin)
    hes = df.assign(E_hes=np.where(df.cluster == "HE-Shear", df.Energy, 0))
    g = hes.groupby(["test_type", "filler_pct"])
    src = pd.DataFrame({"HE_shear_hit_share_%": g.apply(lambda x: (x.cluster == "HE-Shear").mean() * 100),
                        "HE_shear_energy_share_%": g.E_hes.sum() / g.Energy.sum() * 100}).round(2)
    src.to_csv(RES / "15_source_study_comparison.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    fl = [0, 1, 3, 5]; w = 0.36
    for i, (tt, col, lab) in enumerate((("CAI", "#eda100", "Impacted (CAI)"),
                                        ("PureComp", "#2a78d6", "Non-impacted (pure compression)"))):
        v = src.loc[tt, "HE_shear_energy_share_%"].reindex(fl).values
        xs = np.arange(4) + (i - 0.5) * w
        ax.bar(xs, v, width=w - 0.03, color=col, label=lab)
        for x, val in zip(xs, v):
            ax.text(x, val + 1, f"{val:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(4), [f"{f} wt%" for f in fl]); ax.set_xlabel("Cenosphere content")
    ax.set_ylabel("Energy share of HE-shear (%)"); ax.set_ylim(0, 75); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Delamination-type (HE-shear) energy share by filler content", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "12_source_study_comparison.png"); plt.close(fig)

    # Step 5 - comparison with manual peak-frequency bands
    manual = df.damage_mode != "Unclassified"
    comp = pd.DataFrame([{
        "hits_classified_by_manual_bands": int(manual.sum()),
        "hits_outside_all_bands": int((~manual).sum()),
        "hits_assigned_by_clustering": len(df),
        "ARI_vs_manual_bands": adjusted_rand_score(df.damage_mode[manual], df.cluster[manual]),
        "AMI_vs_manual_bands": adjusted_mutual_info_score(df.damage_mode[manual], df.cluster[manual]),
    }])
    comp.to_csv(RES / "07_manual_band_comparison.csv", index=False)
    pd.crosstab(df.damage_mode, df.cluster)[ORDER].to_csv(RES / "08_manual_band_vs_cluster_crosstab.csv")
    gap = df[~manual].cluster.value_counts().reindex(ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.barh(ORDER, gap.values, color=[COLORS[c] for c in ORDER], height=0.6)
    for y, v in enumerate(gap.values):
        ax.text(v + 15, y, f"{v:,}", va="center", fontsize=9)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel("AE hits")
    ax.set_title(f"{(~manual).sum():,} hits outside all manual frequency bands, assigned by clustering",
                 loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "07_gap_band_hits_assigned.png"); plt.close(fig)

    # Step 6 - Random Forest deployment model
    y = df.cluster
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.10,
                                      stratify=df.specimen, random_state=SEED)
    k_nn = tune_knn(df[FEATURES].iloc[idx_tr], y.iloc[idx_tr])
    rows = [{"model": m, "validation": "90/10 hold-out (mentor split)", "fold": "all",
             "accuracy": a, "macro_f1": f}
            for m, (a, f) in evaluate_fold(df, Z, idx_tr, idx_te, k_nn).items()]
    for tr, te in LeaveOneGroupOut().split(Z, groups=df.specimen):
        for m, (a, f) in evaluate_fold(df, Z, tr, te, k_nn).items():
            rows.append({"model": m, "validation": "Leave-one-specimen-out",
                         "fold": df.specimen.iloc[te[0]], "accuracy": a, "macro_f1": f})
    allres = pd.DataFrame(rows)
    allres.round(4).to_csv(RES / "09_model_validation_all_folds.csv", index=False)
    summary = allres.groupby(["model", "validation"], sort=False)[["accuracy", "macro_f1"]].mean().reset_index()
    summary.round(4).to_csv(RES / "10_rf_vs_knn_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.4, 3))
    labels = ["90/10 hold-out", "Leave-one-specimen-out"]
    models = summary.model.unique(); w = 0.36
    for i, (m, col) in enumerate(zip(models, ["#2a78d6", "#eb6834"])):
        vals = summary[summary.model == m].accuracy.values * 100
        xs = np.arange(2) + (i - 0.5) * w
        ax.bar(xs, vals, width=w - 0.03, color=col, label=m)
        for x, v in zip(xs, vals):
            ax.text(x, v + 2, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(range(2), labels); ax.set_ylim(0, 115); ax.set_yticks(range(0, 101, 20)); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Accuracy (%)"); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax.set_title("Benchmark: Random Forest vs kNN (same features, same splits)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "10_rf_vs_knn_benchmark.png"); plt.close(fig)

    # Final deployment model on the 90 % training split, report on the 10 % test split
    model = rf().fit(df[FEATURES].iloc[idx_tr], y.iloc[idx_tr])
    pred = model.predict(df[FEATURES].iloc[idx_te])
    with open(RES / "11_rf_classification_report_90_10.txt", "w") as fh:
        fh.write(classification_report(y.iloc[idx_te], pred, digits=3))
    cm = pd.DataFrame(confusion_matrix(y.iloc[idx_te], pred, labels=ORDER), index=ORDER, columns=ORDER)
    cm.to_csv(RES / "12_rf_confusion_matrix_90_10.csv")

    # Prediction confidence (max class probability)
    conf = model.predict_proba(df[FEATURES].iloc[idx_te]).max(axis=1)
    ok = pred == y.iloc[idx_te].values
    sure = conf >= CONF_THRESHOLD
    cres = pd.DataFrame([
        {"group": f"Confident (>= {CONF_THRESHOLD:.0%})", "hits": int(sure.sum()),
         "share_%": sure.mean() * 100, "accuracy_%": ok[sure].mean() * 100},
        {"group": f"Uncertain (< {CONF_THRESHOLD:.0%})", "hits": int((~sure).sum()),
         "share_%": (~sure).mean() * 100, "accuracy_%": ok[~sure].mean() * 100 if (~sure).any() else np.nan},
    ])
    cres.round(2).to_csv(RES / "14_prediction_confidence.csv", index=False)
    print(cres.round(1).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.4, 3))
    bins = np.linspace(0.25, 1, 31)
    ax.hist([conf[ok], conf[~ok]], bins=bins, stacked=True, color=["#2a78d6", "#eb6834"],
            label=["Correct", "Wrong"], edgecolor="white", linewidth=1)
    ax.axvline(CONF_THRESHOLD, color="#52514e", ls="--", lw=1.2)
    ax.text(CONF_THRESHOLD - 0.01, ax.get_ylim()[1] * 0.9, "uncertain | confident", ha="center", fontsize=8,
            color="#52514e", bbox=dict(fc="white", ec="none"))
    ax.set_yscale("log"); ax.set_xlabel("Prediction confidence"); ax.set_ylabel("Test hits (log scale)")
    ax.grid(axis="x", visible=False); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Wrong predictions concentrate at low confidence (10% test set)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "11_prediction_confidence.png"); plt.close(fig)

    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    imp.round(4).to_csv(RES / "13_rf_feature_importance.csv", header=["importance"])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(imp.index, imp.values, color="#2a78d6", height=0.6)
    for yv, v in enumerate(imp.values):
        ax.text(v + 0.004, yv, f"{v:.3f}", va="center", fontsize=8)
    ax.grid(axis="y", visible=False); ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance (damage-mechanism model)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "08_rf_feature_importance.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.imshow(cm.values, cmap="Blues"); ax.grid(False)
    ax.set_xticks(range(K), ORDER, rotation=30, ha="right"); ax.set_yticks(range(K), ORDER)
    for i in range(K):
        for j in range(K):
            v = cm.values[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                    color="white" if v > cm.values.max() / 2 else "#0b0b0b")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix, 10% test set", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "09_rf_confusion_matrix.png"); plt.close(fig)

    joblib.dump({"model": model, "features": FEATURES, "classes": ORDER,
                 "descriptions": LONG, "conf_threshold": CONF_THRESHOLD}, MOD / "ae_cluster_model.joblib")
    df.drop(columns=["PC1", "PC2"]).to_csv(RES / "ae_hits_with_clusters.csv", index=False)
    print("Done. Figures ->", FIG, "| Results ->", RES)


if __name__ == "__main__":
    main()
